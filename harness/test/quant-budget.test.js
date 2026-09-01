import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compareVectors } from "../lib/metrics.js";
import { loadEpsilon, loadGoldens } from "../lib/goldens.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDENS = join(HERE, "..", "goldens");

describe("F32 quant budget (derived, not hand-picked)", () => {
  const budget = JSON.parse(readFileSync(join(GOLDENS, "quant-budget.json"), "utf8"));
  const f16 = JSON.parse(readFileSync(join(GOLDENS, "vectors-f16.json"), "utf8"));
  const q4 = loadGoldens();
  const eps = loadEpsilon();
  const f16ById = new Map(f16.items.map((it) => [it.id, it]));

  it("does not rewrite the Q4-vs-Q4 epsilon floor", () => {
    assert.equal(eps.epsilon, 1e-6);
    assert.equal(eps.epsilon_abs, 1e-5);
  });

  it("schema and pin are the original F16 GGUF, not dequantized Q4", () => {
    assert.equal(budget.schema, "milton.quant-budget/1");
    assert.equal(budget.ref_f32.gguf_file_type, "MOSTLY_F16");
    assert.equal(
      budget.ref_f32.gguf_sha256,
      "9e661465bea62ac3e494b31f3a3fccdad76f89140a75097399f441f4907da99b",
    );
    assert.match(budget.ref_f32.source, /NOT dequantized Q4_K_M/);
    assert.equal(budget.safety_factor, 3);
    assert.equal(budget.ratio_max, 1.5);
    assert.deepEqual(budget.pending_excluded, []);
    assert.equal(budget.n_gated, 18);
  });

  it("per-case budget is cos_dist(ref_f32, q_llama); gate = max(gated)×3", () => {
    assert.equal(budget.per_case.length, q4.items.length);
    let maxGated = 0;
    for (const q of q4.items) {
      const r = f16ById.get(q.id);
      assert.ok(r, q.id);
      const cmp = compareVectors(q.vector, r.vector, { epsilon: 0, epsilonAbs: 0 });
      const row = budget.per_case.find((p) => p.id === q.id);
      assert.ok(row, q.id);
      assert.ok(Math.abs(row.quant_budget_cos_dist - cmp.cos_dist) < 1e-12, q.id);
      if (!budget.pending_excluded.includes(q.id)) {
        maxGated = Math.max(maxGated, cmp.cos_dist);
      }
    }
    assert.ok(Math.abs(budget.max_quant_budget_cos_dist_gated - maxGated) < 1e-12);
    assert.ok(Math.abs(budget.gate_cos_dist - maxGated * 3) < 1e-12);
  });
});
