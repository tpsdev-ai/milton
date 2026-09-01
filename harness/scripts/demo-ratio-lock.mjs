#!/usr/bin/env node
/**
 * Regression lock for #18: a synthetic case under the loose absolute
 * but > 1.5× its own quant budget. Old gate (absolute only) PASSES;
 * hardened gate FAILS. Exit 0 only when that red-before / green-after
 * lock holds.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { f32GatePass } from "../lib/f32-gate.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const budget = JSON.parse(
  readFileSync(join(HERE, "..", "goldens", "quant-budget.json"), "utf8"),
);

const synthetic = {
  id: "synthetic-under-absolute-over-ratio",
  quant_budget_cos_dist: 0.05,
  cos_dist: 0.10,
};

const oldPass = synthetic.cos_dist <= budget.gate_cos_dist;
const hardened = f32GatePass({
  cos_dist: synthetic.cos_dist,
  quant_budget_cos_dist: synthetic.quant_budget_cos_dist,
  gate_cos_dist: budget.gate_cos_dist,
  ratio_max: budget.ratio_max,
});

const report = {
  schema: "milton.f32-ratio-lock/1",
  gate_cos_dist: budget.gate_cos_dist,
  ratio_max: budget.ratio_max,
  synthetic,
  ratio: hardened.ratio,
  old_absolute_only: { pass: oldPass },
  hardened: {
    within_absolute: hardened.within_absolute,
    within_ratio: hardened.within_ratio,
    pass: hardened.pass,
  },
  red_before: oldPass === true,
  green_after: hardened.pass === false,
  lock_holds: oldPass === true && hardened.pass === false,
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.lock_holds) process.exit(1);
