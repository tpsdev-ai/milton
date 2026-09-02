import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { crossoverThreshold } from "../../src/q4k-calibrate.js";

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
});
