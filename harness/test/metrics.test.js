import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { cosine, maxAbsDiff, compareVectors, l2Normalize } from "../lib/metrics.js";

describe("gate metrics", () => {
  it("identical vectors pass both legs", () => {
    const a = new Float32Array([0.6, 0.8, 0]);
    const r = compareVectors(a, a, { epsilon: 1e-6, epsilonAbs: 1e-5 });
    assert.equal(r.pass, true);
    assert.ok(Math.abs(r.cosine - 1) < 1e-12);
    assert.equal(r.max_abs, 0);
  });

  it("cosine alone is not enough — scale bug fails abs", () => {
    const a = new Float32Array([0.6, 0.8]);
    const b = new Float32Array([1.2, 1.6]);
    assert.ok(cosine(a, b) > 0.999);
    const r = compareVectors(a, b, { epsilon: 1e-3, epsilonAbs: 1e-3 });
    assert.equal(r.pass, false);
    assert.match(r.reason, /max_abs/);
  });

  it("dim mismatch fails closed", () => {
    const r = compareVectors(new Float32Array([1, 0]), new Float32Array([1, 0, 0]), {
      epsilon: 1,
      epsilonAbs: 1,
    });
    assert.equal(r.pass, false);
    assert.match(r.reason, /dim_mismatch/);
  });

  it("l2Normalize is a no-op on a zero vector", () => {
    const z = l2Normalize(new Float32Array([0, 0, 0]));
    assert.deepEqual([...z], [0, 0, 0]);
  });

  it("maxAbsDiff matches the worst dimension", () => {
    assert.equal(maxAbsDiff([0, 0, 0], [0, 0.5, -0.25]), 0.5);
  });
});
