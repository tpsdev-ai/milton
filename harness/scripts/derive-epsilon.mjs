#!/usr/bin/env node
/**
 * Derive EPSILON / EPSILON_ABS from the reference's own run-to-run delta.
 *
 * Run the pinned llama.cpp path twice on the full corpus, measure max
 * cosine-distance and max per-dim abs-diff, then set the gate a small
 * margin above that floor. Never loosen later to pass an embedder.
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { compareVectors } from "../lib/metrics.js";
import { createReferenceEmbedder, resolvePaths } from "../lib/reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const corpus = loadCorpus();
const paths = resolvePaths();
const embed = createReferenceEmbedder(paths);

const run = async () => {
  const out = [];
  for (const c of corpus.cases) {
    out.push({ id: c.id, vector: await embed(c.text, { prefix: c.prefix }) });
  }
  return out;
};

process.stderr.write("reference run A...\n");
const a = await run();
process.stderr.write("reference run B...\n");
const b = await run();

let maxCosDist = 0;
let maxAbs = 0;
const per = [];
for (let i = 0; i < a.length; i++) {
  const cmp = compareVectors(a[i].vector, b[i].vector, { epsilon: 0, epsilonAbs: 0 });
  maxCosDist = Math.max(maxCosDist, cmp.cos_dist);
  maxAbs = Math.max(maxAbs, cmp.max_abs);
  per.push({ id: a[i].id, cos_dist: cmp.cos_dist, max_abs: cmp.max_abs });
}

// Floor is the observed self-delta. Margin: 10× the floor, with a numeric
// floor so a bit-identical pair still yields a real (tight) gate rather than
// 0 — which would make any ulp of future-host drift a false fail, and would
// also make "loosen to pass" indistinguishable from "the number is zero".
const COS_MARGIN = 10;
const ABS_MARGIN = 10;
const COS_MIN = 1e-6;
const ABS_MIN = 1e-5;

const epsilon = Math.max(maxCosDist * COS_MARGIN, COS_MIN);
const epsilonAbs = Math.max(maxAbs * ABS_MARGIN, ABS_MIN);

const record = {
  schema: "milton.epsilon/1",
  epsilon,
  epsilon_abs: epsilonAbs,
  derived_from: {
    method: "reference run-to-run: llama-embedding twice, same GGUF, threads=1, pooling=mean, embd-normalize=2",
    n: a.length,
    observed_max_cos_dist: maxCosDist,
    observed_max_abs: maxAbs,
    cos_margin: COS_MARGIN,
    abs_margin: ABS_MARGIN,
    cos_numeric_floor: COS_MIN,
    abs_numeric_floor: ABS_MIN,
    formula: "epsilon = max(observed_max_cos_dist * 10, 1e-6); epsilon_abs = max(observed_max_abs * 10, 1e-5)",
    per_item: per,
  },
};

writeFileSync(join(HERE, "..", "goldens", "epsilon.json"), `${JSON.stringify(record, null, 2)}\n`);
process.stdout.write(JSON.stringify(record, null, 2) + "\n");
