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
  return { ...JSON.parse(ran.stdout), stderr: ran.stderr || "" };
}

const CLAMP = join(tmpdir(), "milton-thread-clamp-child.mjs");
writeFileSync(
  CLAMP,
  `import { resolveThreadCount, hostParallelism } from ${JSON.stringify(join(ROOT, "src", "wasm-threads.js"))};
const warnings = [];
console.warn = (msg) => { warnings.push(String(msg)); };
const raw = process.env.CLAMP_THREADS;
const env = raw === "__UNSET__" ? {} : { MILTON_THREADS: raw === "__EMPTY__" ? "" : raw };
const first = resolveThreadCount(env);
const second = resolveThreadCount(env);
process.stdout.write(JSON.stringify({
  first,
  second,
  warnings,
  cores: hostParallelism(),
}));
`,
);

function clampRun(raw) {
  const ran = spawnSync(process.execPath, [CLAMP], {
    encoding: "utf8",
    timeout: 10000,
    env: { ...process.env, CLAMP_THREADS: raw },
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
    assert.equal(
      got.stderr.includes("MILTON_THREADS="),
      false,
      `in-range MILTON_THREADS=1 must not warn, got ${JSON.stringify(got.stderr)}`,
    );
    assert.equal(got.artifact, "single");
    assert.equal(got.threads, 1);
    assert.deepEqual(Object.keys(got.report).sort(), [
      "artifact",
      "availableParallelism",
      "sabAvailable",
      "wasm",
      "workers",
    ]);
    assert.equal(got.report.wasm, "milton_relaxed_bg.wasm");
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
    assert.equal(got.report.wasm, "milton_threads_relaxed_bg.wasm");
    assert.equal(got.report.sabAvailable, true);
    assert.equal(got.report.availableParallelism, cores);
    assert.equal(
      got.stderr.includes("MILTON_THREADS="),
      false,
      `in-range MILTON_THREADS=${want} must not warn, got ${JSON.stringify(got.stderr)}`,
    );
  });
});

describe("MILTON_THREADS out-of-range clamp warn", () => {
  it("env=abc (and 0 / negative / non-finite) clamps to 1 and warns once", () => {
    for (const raw of ["abc", "0", "-3", "NaN", "Infinity", "-Infinity"]) {
      const got = clampRun(raw);
      assert.equal(got.first, 1, raw);
      assert.equal(got.second, 1, raw);
      assert.equal(got.warnings.length, 1, raw);
      assert.ok(
        got.warnings[0].includes(`MILTON_THREADS=${raw}`),
        `expected warn naming MILTON_THREADS=${raw}, got ${JSON.stringify(got.warnings)}`,
      );
      assert.ok(
        got.warnings[0].includes("using 1"),
        `expected warn naming applied 1, got ${JSON.stringify(got.warnings)}`,
      );
    }
  });

  it("env=9999 clamps to hostParallelism() and warns once", () => {
    const got = clampRun("9999");
    assert.equal(got.first, got.cores);
    assert.equal(got.second, got.cores);
    assert.equal(got.warnings.length, 1);
    assert.ok(
      got.warnings[0].includes("MILTON_THREADS=9999"),
      `expected warn naming MILTON_THREADS=9999, got ${JSON.stringify(got.warnings)}`,
    );
    assert.ok(
      got.warnings[0].includes(`using ${got.cores}`),
      `expected warn naming applied ${got.cores}, got ${JSON.stringify(got.warnings)}`,
    );
  });

  it("unset / empty / in-range values do not warn", () => {
    const cores = hostParallelism();
    for (const raw of ["__UNSET__", "__EMPTY__", "1", String(Math.min(4, cores))]) {
      const got = clampRun(raw);
      assert.equal(got.warnings.length, 0, raw);
      if (raw === "__UNSET__" || raw === "__EMPTY__") {
        assert.equal(got.first, Math.min(4, cores));
      }
    }
  });

  it("env=abc embed reports workers 1 and warns naming the applied value", () => {
    const got = run({ MILTON_THREADS: "abc" });
    assert.equal(got.threads, 1);
    assert.equal(got.report.workers, 1);
    assert.ok(
      got.stderr.includes("MILTON_THREADS=abc"),
      `expected stderr to name MILTON_THREADS=abc, got ${JSON.stringify(got.stderr)}`,
    );
    assert.ok(
      got.stderr.includes("using 1"),
      `expected stderr to name applied 1, got ${JSON.stringify(got.stderr)}`,
    );
    assert.equal((got.stderr.match(/MILTON_THREADS=/g) || []).length, 1);
  });

  it("env=9999 embed reports workers = cores and warns naming the applied value", () => {
    const cores = hostParallelism();
    const got = run({ MILTON_THREADS: "9999" });
    assert.equal(got.threads, cores);
    assert.equal(got.report.workers, cores);
    assert.ok(
      got.stderr.includes("MILTON_THREADS=9999"),
      `expected stderr to name MILTON_THREADS=9999, got ${JSON.stringify(got.stderr)}`,
    );
    assert.ok(
      got.stderr.includes(`using ${cores}`),
      `expected stderr to name applied ${cores}, got ${JSON.stringify(got.stderr)}`,
    );
    assert.equal((got.stderr.match(/MILTON_THREADS=/g) || []).length, 1);
  });
});
