import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { f32GatePass } from "../lib/f32-gate.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const GOLDENS = join(HERE, "..", "goldens");

describe("F32 gate: within_absolute AND ratio <= ratio_max", () => {
  const budget = JSON.parse(readFileSync(join(GOLDENS, "quant-budget.json"), "utf8"));
  const eps = JSON.parse(readFileSync(join(GOLDENS, "epsilon.json"), "utf8"));

  it("pins ratio_max = 1.5 next to safety_factor (one source)", () => {
    assert.equal(budget.ratio_max, 1.5);
    assert.equal(budget.safety_factor, 3);
    assert.ok(budget.gate_cos_dist > 0.308 && budget.gate_cos_dist < 0.309);
  });

  it("does not loosen the Q4-vs-Q4 epsilon floor", () => {
    assert.equal(eps.epsilon, 1e-6);
    assert.equal(eps.epsilon_abs, 1e-5);
  });

  it("synthetic under-absolute over-ratio FAILS (old absolute-only would PASS)", () => {
    // Hole #18 closes: sit under 0.3081 but > 1.5× this case's own budget.
    const quant_budget_cos_dist = 0.05;
    const cos_dist = 0.10; // 2.0× budget, still < gate_cos_dist
    const oldAbsoluteOnly = cos_dist <= budget.gate_cos_dist;
    assert.equal(oldAbsoluteOnly, true, "old absolute-only gate would have passed — this is the hole");
    assert.ok(cos_dist / quant_budget_cos_dist > budget.ratio_max);

    const r = f32GatePass({
      cos_dist,
      quant_budget_cos_dist,
      gate_cos_dist: budget.gate_cos_dist,
      ratio_max: budget.ratio_max,
    });
    assert.equal(r.within_absolute, true);
    assert.equal(r.within_ratio, false);
    assert.equal(r.pass, false);
    assert.ok(Math.abs(r.ratio - 2.0) < 1e-12);
  });

  it("recorded gated ratios 0.90–1.08 still pass", () => {
    const qb = 0.05;
    for (const ratio of [0.90, 1.00, 1.08]) {
      const r = f32GatePass({
        cos_dist: ratio * qb,
        quant_budget_cos_dist: qb,
        gate_cos_dist: budget.gate_cos_dist,
        ratio_max: budget.ratio_max,
      });
      assert.equal(r.pass, true, `ratio ${ratio}`);
      assert.equal(r.within_absolute, true);
      assert.equal(r.within_ratio, true);
    }
  });

  it("tight-tier empty-none / short-hello-none 1e-6 floor is tighter than any ratio", () => {
    const emptyNone = budget.per_case.find((p) => p.id === "empty-none");
    const shortHelloNone = budget.per_case.find((p) => p.id === "short-hello-none");
    assert.ok(emptyNone && shortHelloNone);
    for (const row of [emptyNone, shortHelloNone]) {
      const r = f32GatePass({
        cos_dist: 1e-6,
        quant_budget_cos_dist: row.quant_budget_cos_dist,
        gate_cos_dist: budget.gate_cos_dist,
        ratio_max: budget.ratio_max,
      });
      assert.equal(r.pass, true, row.id);
      assert.ok(r.ratio < 1e-4, row.id);
    }
  });

  it("both gate implementations read ratio_max and call the shared predicate", () => {
    const js = readFileSync(join(ROOT, "harness/scripts/discriminate-f32.mjs"), "utf8");
    const rs = readFileSync(join(ROOT, "crate/src/bin/embed-gate.rs"), "utf8");
    const derive = readFileSync(join(ROOT, "harness/scripts/derive-quant-budget.mjs"), "utf8");
    assert.match(js, /f32GatePass/);
    assert.match(js, /ratio_max/);
    assert.match(rs, /f32_gate_pass/);
    assert.match(rs, /ratio_max/);
    assert.match(derive, /RATIO_MAX = 1\.5/);
    assert.match(derive, /ratio_max: RATIO_MAX/);
  });

  it("fail-closed when ratio_max is missing or non-positive", () => {
    assert.throws(
      () =>
        f32GatePass({
          cos_dist: 0.01,
          quant_budget_cos_dist: 0.05,
          gate_cos_dist: budget.gate_cos_dist,
          ratio_max: 0,
        }),
      /ratio_max must be positive/,
    );
  });
});
