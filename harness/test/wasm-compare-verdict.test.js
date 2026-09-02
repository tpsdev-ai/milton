import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { judge, loadExpected } from "../scripts/wasm-compare-verdict.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const EXPECTED_PATH = join(HERE, "..", "expected.json");

describe("wasm-compare two-way expected-outcome (flair#1468 shape)", () => {
  it("committed expected.json says fail and names the softmax residual, not RoPE/#27", () => {
    const doc = loadExpected(readFileSync(EXPECTED_PATH, "utf8"), EXPECTED_PATH);
    assert.equal(doc.expected, "fail");
    assert.match(doc.reason, /softmax/i);
    assert.match(doc.reason, /expf|compiler-builtins/i);
    assert.doesNotMatch(doc.reason, /RoPE sinf is the live residual/i);
    assert.ok(Array.isArray(doc.blocked_on));
    assert.ok(doc.blocked_on.includes(25));
    assert.ok(doc.blocked_on.includes(26));
  });

  it("expected fail + observed fail is GREEN", () => {
    const v = judge({
      expected: "fail",
      observed: "fail",
      reason: "softmax residual",
    });
    assert.equal(v.ok, true);
    assert.match(v.summary, /^GREEN:/);
  });

  it("unexpected PASS is RED (must flip expected.json)", () => {
    const v = judge({
      expected: "fail",
      observed: "pass",
      reason: "softmax residual",
    });
    assert.equal(v.ok, false);
    assert.match(v.summary, /unexpected PASS/);
  });

  it("unexpected FAIL is RED when expected is pass", () => {
    const v = judge({
      expected: "pass",
      observed: "fail",
      reason: "should be gone",
    });
    assert.equal(v.ok, false);
    assert.match(v.summary, /unexpected FAIL/);
  });

  it("abort (no receipt) is RED even if expected is fail", () => {
    const v = judge({
      expected: "fail",
      observed: "abort",
      reason: "softmax residual",
    });
    assert.equal(v.ok, false);
    assert.match(v.summary, /aborted/);
  });

  it("fail-closed on a missing reason or bad expected value", () => {
    assert.throws(() => loadExpected(JSON.stringify({ expected: "fail" })), /reason/);
    assert.throws(
      () => loadExpected(JSON.stringify({ expected: "maybe", reason: "x" })),
      /pass" or "fail/,
    );
  });
});
