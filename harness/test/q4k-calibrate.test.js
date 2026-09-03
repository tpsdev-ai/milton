import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { applyQ4kPolicy, crossoverThreshold, WARMUP } from "../../src/q4k-calibrate.js";

describe("Q4_K load-time threshold rule", () => {
  it("never picks all-k when it loses the full tile", () => {
    assert.equal(crossoverThreshold(1, 10, 2, 12), 33);
    assert.equal(crossoverThreshold(1, 10, 1, 10), 33);
  });

  it("picks threshold 1 when all-k wins n=1 and n=32", () => {
    assert.equal(crossoverThreshold(2, 20, 1, 10), 1);
  });

  it("interpolates the crossover when all-k loses n=1 and wins n=32", () => {
    // T_p(n) = 1 + 9*(n-1)/31 ; T_a(n) = 4 + 4*(n-1)/31
    // equal at n = 1 + 31*(1-4)/(4-9) = 1 + 31*3/5 = 19.6 → 20
    assert.equal(crossoverThreshold(1, 10, 4, 8), 20);
  });

  it("keeps WARMUP=100 for a later second variant", () => {
    assert.equal(WARMUP, 100);
  });
});

describe("Q4_K applyQ4kPolicy with all-k not shipped", () => {
  function mockApi(threshold = 33) {
    let force = "auto";
    let t = threshold;
    return {
      q4kSetForce(name) {
        force = name;
      },
      q4kSetThreshold(v) {
        t = v;
      },
      q4kThreshold() {
        return t;
      },
      lastForce: () => force,
    };
  }

  it("auto records threshold 33 and does not time a missing kernel", () => {
    const api = mockApi();
    const report = applyQ4kPolicy(api, {});
    assert.equal(report.mode, "auto");
    assert.equal(report.forced, false);
    assert.equal(report.threshold, 33);
    assert.equal(report.max_abs, null);
    assert.deepEqual(report.shipped_variants, ["perk"]);
    assert.ok(report.cost_ms < 2);
    assert.equal(api.lastForce(), "auto");
  });

  it("force perk is still allowed", () => {
    const api = mockApi(33);
    const report = applyQ4kPolicy(api, { MILTON_Q4K_VARIANT: "perk" });
    assert.equal(report.mode, "perk");
    assert.equal(report.forced, true);
    assert.equal(api.lastForce(), "perk");
  });

  it("force allk fail-closes", () => {
    const api = mockApi();
    assert.throws(
      () => applyQ4kPolicy(api, { MILTON_Q4K_VARIANT: "allk" }),
      /all-k is not shipped/,
    );
  });
});
