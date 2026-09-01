#!/usr/bin/env node
/**
 * Discriminate Milton Q4_K_M against the F16/F32 oracle.
 *
 *   milton_vs_f32 ≈ q_llama_vs_f32  → correct (residual is quantization)
 *   milton_vs_f32 ≫ budget          → real bug
 *
 * Also reports milton vs q_llama (sanity: ~1e-5 once correct, not 1e-2).
 */
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { f32GatePass } from "../lib/f32-gate.js";
import { loadGoldens } from "../lib/goldens.js";
import { compareVectors } from "../lib/metrics.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const F16_PATH = join(HERE, "..", "goldens", "vectors-f16.json");
const BUDGET_PATH = join(HERE, "..", "goldens", "quant-budget.json");
const OUT = join("/opt/cursor/artifacts", "discriminate-f32.json");

const PENDING_EXCLUDED = new Set(); // #15 closed — all 18 gated

const corpus = loadCorpus();
const qLlama = loadGoldens();
const f16 = JSON.parse(readFileSync(F16_PATH, "utf8"));
const budget = JSON.parse(readFileSync(BUDGET_PATH, "utf8"));
if (!(budget.ratio_max > 0)) {
  throw new Error("fail-closed: quant-budget ratio_max must be positive");
}
if (!(budget.gate_cos_dist > 0)) {
  throw new Error("fail-closed: quant-budget gate_cos_dist must be positive");
}
const qById = new Map(qLlama.items.map((it) => [it.id, it]));
const f16ById = new Map(f16.items.map((it) => [it.id, it]));

function miltonEmbedAll() {
  return new Promise((resolve, reject) => {
    const bin = join(ROOT, "crate/target/release/milton-embed");
    const child = spawn(bin, ["--jsonl"], { stdio: ["pipe", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (d) => stdout.push(d));
    child.stderr.on("data", (d) => stderr.push(d));
    child.on("error", reject);
    for (const c of corpus.cases) {
      child.stdin.write(JSON.stringify({ text: c.text, prefix: c.prefix }) + "\n");
    }
    child.stdin.end();
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`milton-embed exited ${code}: ${Buffer.concat(stderr).toString("utf8").slice(-800)}`));
        return;
      }
      const lines = Buffer.concat(stdout).toString("utf8").trim().split("\n");
      const out = [];
      for (let i = 0; i < corpus.cases.length; i++) {
        const parsed = JSON.parse(lines[i]);
        if (parsed.error) {
          reject(new Error(`${corpus.cases[i].id}: ${parsed.error}`));
          return;
        }
        out.push({ id: corpus.cases[i].id, vector: parsed.vector });
      }
      resolve(out);
    });
  });
}

const milton = await miltonEmbedAll();
const rows = [];
let maxMiltonF32Gated = 0;
let maxMiltonQ4Gated = 0;
let overBudget = [];
for (let i = 0; i < milton.length; i++) {
  const id = milton[i].id;
  const pending = PENDING_EXCLUDED.has(id);
  const vsF32 = compareVectors(milton[i].vector, f16ById.get(id).vector, { epsilon: 0, epsilonAbs: 0 });
  const vsQ4 = compareVectors(milton[i].vector, qById.get(id).vector, { epsilon: 0, epsilonAbs: 0 });
  const qBudget = budget.per_case.find((r) => r.id === id);
  const decision = f32GatePass({
    cos_dist: vsF32.cos_dist,
    quant_budget_cos_dist: qBudget.quant_budget_cos_dist,
    gate_cos_dist: budget.gate_cos_dist,
    ratio_max: budget.ratio_max,
  });
  if (!pending) {
    maxMiltonF32Gated = Math.max(maxMiltonF32Gated, vsF32.cos_dist);
    maxMiltonQ4Gated = Math.max(maxMiltonQ4Gated, vsQ4.cos_dist);
    if (!decision.pass) overBudget.push(id);
  }
  rows.push({
    id,
    pending_issue: pending ? 15 : null, // reserved; set is empty after #15
    quant_budget_cos_dist: qBudget.quant_budget_cos_dist,
    milton_vs_f32_cos_dist: vsF32.cos_dist,
    milton_vs_f32_max_abs: vsF32.max_abs,
    milton_vs_q_llama_cos_dist: vsQ4.cos_dist,
    milton_vs_q_llama_max_abs: vsQ4.max_abs,
    ratio_vs_quant_budget: decision.ratio,
    within_absolute: decision.within_absolute,
    within_ratio: decision.within_ratio,
    within_gate: decision.pass,
    milton_head: milton[i].vector.slice(0, 4),
    f32_head: f16ById.get(id).vector.slice(0, 4),
    q_llama_head: qById.get(id).vector.slice(0, 4),
  });
}

const verdict =
  overBudget.length === 0
    ? "correct-within-budget"
    : "bug-over-budget";

const report = {
  schema: "milton.discriminate.f32/1",
  verdict,
  gate_cos_dist: budget.gate_cos_dist,
  safety_factor: budget.safety_factor,
  ratio_max: budget.ratio_max,
  max_quant_budget_cos_dist_gated: budget.max_quant_budget_cos_dist_gated,
  max_milton_vs_f32_cos_dist_gated: maxMiltonF32Gated,
  max_milton_vs_q_llama_cos_dist_gated: maxMiltonQ4Gated,
  over_budget_ids: overBudget,
  pending_excluded: [...PENDING_EXCLUDED],
  cases: rows,
};

writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(JSON.stringify(report, null, 2) + "\n");
