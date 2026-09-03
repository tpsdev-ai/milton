import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { checkReceiptShape, judge, loadExpected, prepareCompareEnv } from "../scripts/wasm-compare-verdict.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const EXPECTED_PATH = join(HERE, "..", "expected.json");

describe("wasm-compare two-way expected-outcome (flair#1468 shape)", () => {
  it("committed expected.json says pass after shared RoPE sin/cos (#27)", () => {
    const doc = loadExpected(readFileSync(EXPECTED_PATH, "utf8"), EXPECTED_PATH);
    assert.equal(doc.expected, "pass");
    assert.match(doc.reason, /sin\/cos|RoPE/i);
    assert.match(doc.reason, /bit-identical/i);
    assert.doesNotMatch(doc.reason, /softmax glibc/i);
    assert.doesNotMatch(doc.reason, /blocked on/i);
    assert.ok(Array.isArray(doc.blocked_on));
    assert.equal(doc.blocked_on.length, 0);
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

  it("threaded receipt shape requires file + pool >1, not a single-thread label", () => {
    const ok = {
      wasm_file: "milton_threads_relaxed_bg.wasm",
      wasm_artifact: "threads",
      wasm_threads: 4,
      max_abs: 0,
      qmatmul_kernel: { kernel: "relaxed", probe: true, forced: false },
      thread_report: {
        artifact: "threads",
        workers: 4,
        wasm: "milton_threads_relaxed_bg.wasm",
      },
    };
    assert.deepEqual(
      checkReceiptShape(ok, {
        artifact: "threads",
        threads: 4,
        kernel: "relaxed",
        maxAbs: 0,
        wasm: "milton_threads_relaxed_bg.wasm",
      }),
      [],
    );
    const labeledSingle = checkReceiptShape(
      {
        wasm_file: "milton_relaxed_bg.wasm",
        wasm_artifact: "single",
        wasm_threads: 1,
        max_abs: 0,
        qmatmul_kernel: { kernel: "relaxed", probe: true, forced: false },
        thread_report: { artifact: "single", workers: 1, wasm: "milton_relaxed_bg.wasm" },
      },
      {
        artifact: "threads",
        threads: 4,
        kernel: "relaxed",
        maxAbs: 0,
        wasm: "milton_threads_relaxed_bg.wasm",
      },
    );
    assert.ok(labeledSingle.length >= 4);
    assert.match(labeledSingle.join("\n"), /wasm_artifact expected threads/);
    assert.match(labeledSingle.join("\n"), /thread_report.workers expected >1/);
    assert.match(labeledSingle.join("\n"), /wasm_file expected milton_threads_relaxed_bg.wasm/);
  });

  it("prepareCompareEnv overrides inherited MILTON_WASM_THREADS=0 on the threads lane", () => {
    const env = prepareCompareEnv(
      { MILTON_WASM_THREADS: "0", MILTON_THREADS: "1", OTHER: "keep" },
      { artifact: "threads", threads: 4 },
    );
    assert.equal(env.MILTON_WASM_THREADS, "1");
    assert.equal(env.MILTON_THREADS, "4");
    assert.equal(env.OTHER, "keep");
  });
});
