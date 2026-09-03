import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { applyQ4kPolicy, crossoverThreshold, SAMPLES, WARMUP } from "../../src/q4k-calibrate.js";

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

/** Scripted clock: each timed call (now / fn / now) consumes one duration. */
function scriptedNow(durations) {
  let t = 0;
  let i = 0;
  let pending = null;
  return () => {
    if (pending == null) {
      pending = durations[Math.min(i, durations.length - 1)];
      i += 1;
      return t;
    }
    t += pending;
    pending = null;
    return t;
  };
}

/** perk32, perk1, bprime32, bprime1 — WARMUP+SAMPLES each. */
function durationsFor(perk32, perk1, bprime32, bprime1) {
  const n = WARMUP + SAMPLES;
  return [
    ...Array(n).fill(perk32),
    ...Array(n).fill(perk1),
    ...Array(n).fill(bprime32),
    ...Array(n).fill(bprime1),
  ];
}

describe("Q4_K applyQ4kPolicy with perk + (b′) shipped", () => {
  function mockApi(threshold = 33) {
    let force = "auto";
    let t = threshold;
    const perkCalls = { 1: 0, 32: 0 };
    const bprimeCalls = { 1: 0, 32: 0 };
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
      q4kRunPerk(n) {
        perkCalls[n] = (perkCalls[n] || 0) + 1;
      },
      q4kRunBprime(n) {
        bprimeCalls[n] = (bprimeCalls[n] || 0) + 1;
      },
      lastForce: () => force,
      perkCalls,
      bprimeCalls,
      // legacy alias used by older assertions
      get calls() {
        return perkCalls;
      },
    };
  }

  it("auto times perk and bprime (WARMUP+SAMPLES at n=32 and n=1)", () => {
    const api = mockApi();
    api.now = scriptedNow(durationsFor(10, 1, 12, 2));
    const report = applyQ4kPolicy(api, {});
    assert.equal(report.mode, "auto");
    assert.equal(report.forced, false);
    assert.equal(report.max_abs, null);
    assert.deepEqual(report.shipped_variants, ["perk", "bprime"]);
    assert.equal(api.lastForce(), "auto");
    assert.equal(api.perkCalls[32], WARMUP + SAMPLES);
    assert.equal(api.perkCalls[1], WARMUP + SAMPLES);
    assert.equal(api.bprimeCalls[32], WARMUP + SAMPLES);
    assert.equal(api.bprimeCalls[1], WARMUP + SAMPLES);
    assert.equal(typeof report.perk_n32_ms, "number");
    assert.equal(typeof report.perk_n32_first5_ms, "number");
    assert.equal(typeof report.perk_n32_last5_ms, "number");
    assert.equal(typeof report.perk_n32_warmup_verdict, "string");
    assert.equal(typeof report.bprime_n32_ms, "number");
    assert.equal(typeof report.bprime_n32_first5_ms, "number");
    assert.equal(typeof report.bprime_n32_last5_ms, "number");
    assert.equal(typeof report.bprime_n32_warmup_verdict, "string");
    assert.equal(report.allk_n32_ms, null);
  });

  it("auto fail-closes to threshold 33 when bprime loses n=32", () => {
    const api = mockApi();
    api.now = scriptedNow(durationsFor(10, 1, 12, 2));
    const report = applyQ4kPolicy(api, {});
    assert.equal(report.threshold, 33);
    assert.equal(report.perk_n32_ms, 10);
    assert.equal(report.bprime_n32_ms, 12);
  });

  it("auto writes the interpolated crossover when bprime wins n=32", () => {
    const api = mockApi();
    // perk1=1, perk32=10, bprime1=4, bprime32=8 → ceil(19.6) = 20
    api.now = scriptedNow(durationsFor(10, 1, 8, 4));
    const report = applyQ4kPolicy(api, {});
    assert.equal(report.threshold, 20);
    assert.equal(report.perk_n1_ms, 1);
    assert.equal(report.perk_n32_ms, 10);
    assert.equal(report.bprime_n1_ms, 4);
    assert.equal(report.bprime_n32_ms, 8);
  });

  it("auto fail-closes when q4kRunPerk is missing (no silent 33)", () => {
    const api = mockApi();
    delete api.q4kRunPerk;
    assert.throws(
      () => applyQ4kPolicy(api, {}),
      /requires q4kRunPerk/,
    );
  });

  it("auto fail-closes when q4kRunBprime is missing (no silent 33)", () => {
    const api = mockApi();
    delete api.q4kRunBprime;
    assert.throws(
      () => applyQ4kPolicy(api, {}),
      /requires q4kRunBprime/,
    );
  });

  it("force perk is still allowed", () => {
    const api = mockApi(33);
    const report = applyQ4kPolicy(api, { MILTON_Q4K_VARIANT: "perk" });
    assert.equal(report.mode, "perk");
    assert.equal(report.forced, true);
    assert.equal(api.lastForce(), "perk");
    assert.equal(api.perkCalls[32], 0);
    assert.equal(api.bprimeCalls[32], 0);
  });

  it("force bprime (and aliases) short-circuits without timing", () => {
    for (const raw of ["bprime", "b-prime", "b_prime", "BPRIME", "b'"]) {
      const api = mockApi(33);
      const report = applyQ4kPolicy(api, { MILTON_Q4K_VARIANT: raw });
      assert.equal(report.mode, "bprime", raw);
      assert.equal(report.forced, true, raw);
      assert.equal(api.lastForce(), "bprime", raw);
      assert.equal(api.perkCalls[32], 0, raw);
      assert.equal(api.bprimeCalls[32], 0, raw);
    }
  });

  it("env=allk loads (no throw), warns, and reports auto not all-k", () => {
    const api = mockApi();
    api.now = scriptedNow(durationsFor(10, 1, 12, 2));
    const warnings = [];
    const orig = console.warn;
    console.warn = (msg) => {
      warnings.push(String(msg));
    };
    let report;
    try {
      report = applyQ4kPolicy(api, { MILTON_Q4K_VARIANT: "allk" });
    } finally {
      console.warn = orig;
    }
    assert.notEqual(report.mode, "allk");
    assert.ok(report.mode === "auto" || report.mode === "perk");
    assert.equal(api.lastForce(), "auto");
    assert.ok(
      warnings.some((w) => w.includes("MILTON_Q4K_VARIANT=allk")),
      `expected warn naming MILTON_Q4K_VARIANT=allk, got ${JSON.stringify(warnings)}`,
    );
  });
});
