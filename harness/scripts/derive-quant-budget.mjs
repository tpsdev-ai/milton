#!/usr/bin/env node
/**
 * Derive the quantization budget from llama.cpp itself:
 *   quant_budget[case] = cos_dist(ref_f32[case], q_llama[case])
 *
 * ref_f32 = llama-embedding on original F16 GGUF (vectors-f16.json)
 * q_llama = llama-embedding on pinned Q4_K_M (vectors.json)
 *
 * Gate tolerance = max(gated cases) × SAFETY (2–3×). Pending #15
 * (unicode-nfd, newlines-tabs) stay in the table but are excluded
 * from the max. Does not rewrite epsilon.json.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compareVectors } from "../lib/metrics.js";
import { loadGoldens } from "../lib/goldens.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "goldens", "quant-budget.json");
const F16_PATH = join(HERE, "..", "goldens", "vectors-f16.json");
const PIN_F16 = join(HERE, "..", "goldens", "pin-f16.json");

const PENDING_15 = new Set(["unicode-nfd", "newlines-tabs"]);
const SAFETY = 3;

const qLlama = loadGoldens();
const f16 = JSON.parse(readFileSync(F16_PATH, "utf8"));
if (!f16 || !Array.isArray(f16.items)) {
  throw new Error(`invalid F16 goldens at ${F16_PATH}`);
}
const pinF16 = JSON.parse(readFileSync(PIN_F16, "utf8"));
const f16ById = new Map(f16.items.map((it) => [it.id, it]));

const per = [];
let maxGated = 0;
let maxAll = 0;
let maxAbsGated = 0;
for (const q of qLlama.items) {
  const r = f16ById.get(q.id);
  if (!r) throw new Error(`missing F16 vector for ${q.id}`);
  const cmp = compareVectors(q.vector, r.vector, { epsilon: 0, epsilonAbs: 0 });
  const row = {
    id: q.id,
    prefix: q.prefix,
    quant_budget_cos_dist: cmp.cos_dist,
    quant_budget_max_abs: cmp.max_abs,
    pending_issue: PENDING_15.has(q.id) ? 15 : null,
  };
  per.push(row);
  maxAll = Math.max(maxAll, cmp.cos_dist);
  if (!PENDING_15.has(q.id)) {
    maxGated = Math.max(maxGated, cmp.cos_dist);
    maxAbsGated = Math.max(maxAbsGated, cmp.max_abs);
  }
}

const gate_cos = maxGated * SAFETY;
const record = {
  schema: "milton.quant-budget/1",
  note: "llama.cpp's own quantization error: cos_dist(llama-embedding F16, llama-embedding Q4_K_M). Gate = max(gated)×3. epsilon.json is unchanged (Q4-vs-Q4 run-to-run floor).",
  safety_factor: SAFETY,
  pending_excluded: [...PENDING_15],
  n: per.length,
  n_gated: per.length - PENDING_15.size,
  max_quant_budget_cos_dist_all: maxAll,
  max_quant_budget_cos_dist_gated: maxGated,
  max_quant_budget_max_abs_gated: maxAbsGated,
  gate_cos_dist: gate_cos,
  formula: "quant_budget[case] = cos_dist(ref_f32, q_llama); gate_cos_dist = max(gated)×3",
  ref_f32: {
    gguf_file: pinF16.gguf_file,
    gguf_sha256: pinF16.gguf_sha256,
    gguf_file_type: pinF16.gguf_file_type,
    llamacpp_commit: pinF16.llamacpp_commit,
    source: pinF16.gguf_source,
  },
  q_llama: {
    gguf_file: "nomic-embed-text-v1.5.Q4_K_M.gguf",
    vectors: "harness/goldens/vectors.json",
  },
  per_case: per,
};

writeFileSync(OUT, `${JSON.stringify(record, null, 2)}\n`);
process.stdout.write(JSON.stringify(record, null, 2) + "\n");
