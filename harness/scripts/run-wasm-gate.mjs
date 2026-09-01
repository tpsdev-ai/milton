#!/usr/bin/env node
/**
 * WASM golden-vector + F32 ratio gate. Same predicates as embed-gate.rs.
 * Does not rewrite epsilon.json / quant-budget.json.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { f32GatePass } from "../lib/f32-gate.js";
import { loadEpsilon, loadGoldens } from "../lib/goldens.js";
import { compareVectors } from "../lib/metrics.js";
import { embed } from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const F16_PATH = join(HERE, "..", "goldens", "vectors-f16.json");
const BUDGET_PATH = join(HERE, "..", "goldens", "quant-budget.json");

const corpus = loadCorpus();
const qLlama = loadGoldens();
const f16 = JSON.parse(readFileSync(F16_PATH, "utf8"));
const budget = JSON.parse(readFileSync(BUDGET_PATH, "utf8"));
const eps = loadEpsilon();
const qById = new Map(qLlama.items.map((it) => [it.id, it]));
const f16ById = new Map(f16.items.map((it) => [it.id, it]));
const pending = new Set(budget.pending_excluded ?? []);
const q4Lock = new Set(["empty-none", "short-hello-none"]);

if (!(budget.ratio_max > 0) || !(budget.gate_cos_dist > 0)) {
  throw new Error("fail-closed: quant-budget ratio_max / gate_cos_dist must be positive");
}

const gated = [];
const failures = [];
const pendingRows = [];
let maxCos = 0;
let sumCos = 0;
let maxAbs = 0;
let maxQ4 = 0;
let minRatio = Infinity;
let maxRatio = 0;
let nGated = 0;

for (const c of corpus.cases) {
  const refF32 = f16ById.get(c.id);
  const q4 = qById.get(c.id);
  const caseBudget = budget.per_case.find((p) => p.id === c.id);
  if (!refF32) {
    failures.push({ id: c.id, reason: "missing_f32_golden" });
    continue;
  }
  let got;
  try {
    got = await embed(c.text, { prefix: c.prefix });
  } catch (err) {
    failures.push({ id: c.id, reason: `embed_threw:${err.message}` });
    continue;
  }
  const vsF32 = compareVectors(got, refF32.vector, {
    epsilon: budget.gate_cos_dist,
    epsilonAbs: Number.POSITIVE_INFINITY,
  });
  const vsQ4 = q4
    ? compareVectors(got, q4.vector, { epsilon: eps.epsilon, epsilonAbs: eps.epsilon_abs })
    : null;
  const qb = caseBudget?.quant_budget_cos_dist ?? 0;
  const decision = f32GatePass({
    cos_dist: vsF32.cos_dist,
    quant_budget_cos_dist: qb,
    gate_cos_dist: budget.gate_cos_dist,
    ratio_max: budget.ratio_max,
  });
  const isPending = pending.has(c.id);
  const lockFail = q4Lock.has(c.id) && vsQ4 && !vsQ4.pass;
  const row = {
    id: c.id,
    prefix: c.prefix,
    quant_budget_cos_dist: qb,
    milton_vs_f32_cos_dist: vsF32.cos_dist,
    milton_vs_f32_max_abs: vsF32.max_abs,
    milton_vs_q_llama_cos_dist: vsQ4?.cos_dist ?? null,
    milton_vs_q_llama_max_abs: vsQ4?.max_abs ?? null,
    ratio_vs_quant_budget: decision.ratio,
    within_absolute: decision.within_absolute,
    within_ratio: decision.within_ratio,
  };
  if (isPending) {
    pendingRows.push(row);
    continue;
  }
  nGated += 1;
  if (vsF32.cos_dist > maxCos) maxCos = vsF32.cos_dist;
  if (Number.isFinite(vsF32.max_abs) && vsF32.max_abs > maxAbs) maxAbs = vsF32.max_abs;
  sumCos += Number.isFinite(vsF32.cos_dist) ? vsF32.cos_dist : 1;
  if (vsQ4 && vsQ4.cos_dist > maxQ4) maxQ4 = vsQ4.cos_dist;
  if (Number.isFinite(decision.ratio) && decision.ratio < minRatio) minRatio = decision.ratio;
  if (Number.isFinite(decision.ratio) && decision.ratio > maxRatio) maxRatio = decision.ratio;
  gated.push(row);
  if (!decision.pass || lockFail) {
    const reason = [];
    if (!decision.within_absolute) {
      reason.push(`cos_dist=${vsF32.cos_dist} > gate_cos_dist=${budget.gate_cos_dist}`);
    }
    if (!decision.within_ratio) {
      reason.push(`ratio=${decision.ratio} > ratio_max=${budget.ratio_max}`);
    }
    if (lockFail) reason.push(`q4_lock:${vsQ4.reason}`);
    failures.push({ ...row, reason: reason.join(",") });
  }
}

const receipt = {
  schema: "milton.embed.receipt/2",
  backend: "wasm-simd",
  oracle: "ref_f32",
  result: failures.length === 0 ? "pass" : "fail",
  n: corpus.cases.length,
  n_gated: nGated,
  failed: failures.length,
  max_cos_dist: maxCos,
  mean_cos_dist: nGated > 0 ? sumCos / nGated : 0,
  max_abs: maxAbs,
  max_milton_vs_q_llama_cos_dist: maxQ4,
  gate_cos_dist: budget.gate_cos_dist,
  ratio_max: budget.ratio_max,
  min_ratio_gated: Number.isFinite(minRatio) ? minRatio : 0,
  max_ratio_gated: maxRatio,
  q4_epsilon_unchanged: { epsilon: eps.epsilon, epsilon_abs: eps.epsilon_abs },
  q4_lock: [...q4Lock],
  pending_excluded: budget.pending_excluded,
  failures,
  gated,
  pending: pendingRows,
};

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(join(ROOT, "harness", "receipts", "wasm-gate.json"), `${JSON.stringify(receipt, null, 2)}\n`);
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
if (receipt.result === "fail") process.exitCode = 1;
