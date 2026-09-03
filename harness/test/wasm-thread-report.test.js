import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  canUseWasmThreads,
  hostParallelism,
  resolveThreadCount,
  sabAvailable,
} from "../../src/wasm-threads.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const INDEX = join(ROOT, "src", "index.js");
const CHILD = join(tmpdir(), "milton-thread-report-child.mjs");
writeFileSync(
  CHILD,
  `import { embed, lastThreadReport, lastThreadCount, lastWasmArtifact } from ${JSON.stringify(INDEX)};
await embed("hello", { prefix: "document" });
process.stdout.write(JSON.stringify({
  report: lastThreadReport,
  artifact: lastWasmArtifact,
  threads: lastThreadCount,
}));
process.exit(0);
`,
);
const PEEK = join(tmpdir(), "milton-thread-report-peek.mjs");
writeFileSync(
  PEEK,
  `import { lastThreadReport } from ${JSON.stringify(INDEX)};
process.stdout.write(JSON.stringify(lastThreadReport));
process.exit(0);
`,
);

function run(env) {
  const ran = spawnSync(process.execPath, [CHILD], {
    encoding: "utf8",
    timeout: 60000,
    env: { ...process.env, ...env },
  });
  assert.equal(ran.status, 0, ran.stderr || ran.stdout);
  return JSON.parse(ran.stdout);
}

describe("lastThreadReport + MILTON_THREADS=1", () => {
  it("is null before any load (same grain as lastQ4kCalibration)", () => {
    const ran = spawnSync(process.execPath, [PEEK], {
      encoding: "utf8",
      timeout: 10000,
      env: { ...process.env },
    });
    assert.equal(ran.status, 0, ran.stderr || ran.stdout);
    assert.equal(JSON.parse(ran.stdout), null);
  });

  it("MILTON_THREADS=1 refuses the threads artifact even when SAB exists", () => {
    assert.equal(sabAvailable(globalThis), true);
    assert.equal(canUseWasmThreads({ MILTON_THREADS: "1" }, globalThis), false);
    assert.equal(canUseWasmThreads({ MILTON_THREADS: "0" }, globalThis), false);
    assert.equal(canUseWasmThreads({ MILTON_WASM_THREADS: "0" }, globalThis), false);
  });

  it("default with SAB still selects threads when the pool would be >1", () => {
    const cores = hostParallelism();
    if (Math.min(4, cores) <= 1) return;
    assert.equal(canUseWasmThreads({}, globalThis), true);
    assert.ok(resolveThreadCount({}) > 1);
  });

  it("MILTON_THREADS=1 embed loads milton_bg.wasm and reports sabAvailable", () => {
    const got = run({ MILTON_THREADS: "1" });
    assert.equal(got.artifact, "single");
    assert.equal(got.threads, 1);
    assert.deepEqual(Object.keys(got.report).sort(), [
      "artifact",
      "availableParallelism",
      "sabAvailable",
      "workers",
    ]);
    assert.equal(got.report.artifact, "single");
    assert.equal(got.report.workers, 1);
    assert.equal(got.report.sabAvailable, true);
    assert.equal(got.report.availableParallelism, hostParallelism());
  });

  it("auto embed loads the threads artifact and sizes the pool", () => {
    const cores = hostParallelism();
    const want = Math.min(4, cores);
    if (want <= 1) return;
    const got = run({ MILTON_THREADS: String(want) });
    assert.equal(got.artifact, "threads");
    assert.equal(got.threads, want);
    assert.equal(got.report.artifact, "threads");
    assert.equal(got.report.workers, want);
    assert.equal(got.report.sabAvailable, true);
    assert.equal(got.report.availableParallelism, cores);
  });
});
