import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { canUseWasmThreads, resolveThreadCount } from "../../src/wasm-threads.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const SCRIPT = join(ROOT, "harness", "scripts", "wasm-sab-absent.mjs");

describe("SAB-absent loader", () => {
  it("probe is false when SharedArrayBuffer is missing", () => {
    assert.equal(
      canUseWasmThreads({ MILTON_WASM_THREADS: "1" }, { Atomics, WebAssembly }),
      false,
    );
  });

  it("probe is false when MILTON_WASM_THREADS=0 even if SAB exists", () => {
    assert.equal(canUseWasmThreads({ MILTON_WASM_THREADS: "0" }, globalThis), false);
  });

  it("default thread count is min(4, cores) and at least 1", () => {
    const n = resolveThreadCount({});
    assert.ok(n >= 1 && n <= 4);
    assert.equal(resolveThreadCount({ MILTON_THREADS: "1" }), 1);
    assert.equal(resolveThreadCount({ MILTON_THREADS: "0" }), 1);
  });

  it("embed picks the single-thread artifact (not an error)", () => {
    const ran = spawnSync(process.execPath, [SCRIPT], {
      cwd: ROOT,
      encoding: "utf8",
    });
    assert.equal(ran.status, 0, ran.stderr || ran.stdout);
    const receipt = JSON.parse(ran.stdout.trim().split("\n").at(-1));
    assert.equal(receipt.result, "pass");
    assert.equal(receipt.artifact, "single");
    assert.equal(receipt.threads, 1);
    assert.equal(receipt.dims, 768);
  });
});
