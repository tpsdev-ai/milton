import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  probeRelaxedSimd,
  RELAXED_DOT_PROBE,
  resolveQmatmulKernel,
  SIMD128_ADD_PROBE,
} from "../../src/relaxed-simd.js";
import { hostParallelism } from "../../src/wasm-threads.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const INDEX = join(ROOT, "src", "index.js");

const CHILD = join(tmpdir(), "milton-relaxed-kernel-child.mjs");
writeFileSync(
  CHILD,
  `import { embed, lastQmatmulKernel, lastThreadReport } from ${JSON.stringify(INDEX)};
await embed("hello", { prefix: "document" });
process.stdout.write(JSON.stringify({ kernel: lastQmatmulKernel, thread: lastThreadReport }));
process.exit(0);
`,
);

const PEEK = join(tmpdir(), "milton-relaxed-kernel-peek.mjs");
writeFileSync(
  PEEK,
  `import { lastQmatmulKernel } from ${JSON.stringify(INDEX)};
process.stdout.write(JSON.stringify(lastQmatmulKernel));
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
  const parsed = JSON.parse(ran.stdout);
  return parsed.kernel ?? parsed;
}

function runFull(env) {
  const ran = spawnSync(process.execPath, [CHILD], {
    encoding: "utf8",
    timeout: 60000,
    env: { ...process.env, ...env },
  });
  assert.equal(ran.status, 0, ran.stderr || ran.stdout);
  return JSON.parse(ran.stdout);
}

describe("relaxed-SIMD probe (not a version sniff)", () => {
  it("validates the relaxed-dot probe on this Node and the SIMD128 control", () => {
    assert.equal(WebAssembly.validate(SIMD128_ADD_PROBE), true);
    assert.equal(probeRelaxedSimd(), WebAssembly.validate(RELAXED_DOT_PROBE));
    assert.equal(probeRelaxedSimd(), true);
  });

  it("resolveQmatmulKernel: default follows the probe; 0 forces simd128; 1 fail-closes if probe is false", () => {
    const auto = resolveQmatmulKernel({}, globalThis);
    assert.equal(auto.kernel, "relaxed");
    assert.equal(auto.probe, true);
    assert.equal(auto.forced, false);
    const off = resolveQmatmulKernel({ MILTON_RELAXED_SIMD: "0" }, globalThis);
    assert.deepEqual(off, { kernel: "simd128", probe: true, forced: true });
    const on = resolveQmatmulKernel({ MILTON_RELAXED_SIMD: "1" }, globalThis);
    assert.deepEqual(on, { kernel: "relaxed", probe: true, forced: true });
  });
});

describe("lastQmatmulKernel", () => {
  it("is null before any load (same grain as lastThreadReport)", () => {
    const ran = spawnSync(process.execPath, [PEEK], {
      encoding: "utf8",
      timeout: 10000,
      env: { ...process.env },
    });
    assert.equal(ran.status, 0, ran.stderr || ran.stdout);
    assert.equal(JSON.parse(ran.stdout), null);
  });

  it("MILTON_THREADS=1 auto-select reports relaxed on Node ≥22", () => {
    const got = run({ MILTON_THREADS: "1" });
    assert.deepEqual(Object.keys(got).sort(), ["forced", "kernel", "probe"]);
    assert.equal(got.kernel, "relaxed");
    assert.equal(got.probe, true);
    assert.equal(got.forced, false);
  });

  it("auto threads reports relaxed kernel and threads artifact together", () => {
    const cores = Math.min(4, hostParallelism());
    if (cores <= 1) return;
    const got = runFull({ MILTON_THREADS: String(cores) });
    assert.equal(got.kernel.kernel, "relaxed");
    assert.equal(got.kernel.probe, true);
    assert.equal(got.thread.artifact, "threads");
    assert.equal(got.thread.workers, cores);
    assert.equal(got.thread.wasm, "milton_threads_relaxed_bg.wasm");
    assert.ok(got.thread.workers > 1);
  });

  it("inherited MILTON_WASM_THREADS=0 without override stays single (the hole Cos found)", () => {
    const got = runFull({ MILTON_THREADS: "4", MILTON_WASM_THREADS: "0" });
    assert.equal(got.thread.artifact, "single");
    assert.equal(got.thread.workers, 1);
    assert.equal(got.thread.wasm, "milton_relaxed_bg.wasm");
  });

  it("MILTON_RELAXED_SIMD=0 reports simd128 with probe still true", () => {
    const got = run({ MILTON_THREADS: "1", MILTON_RELAXED_SIMD: "0" });
    assert.equal(got.kernel, "simd128");
    assert.equal(got.probe, true);
    assert.equal(got.forced, true);
  });

  it("MILTON_RELAXED_SIMD=1 reports relaxed forced", () => {
    const got = run({ MILTON_THREADS: "1", MILTON_RELAXED_SIMD: "1" });
    assert.equal(got.kernel, "relaxed");
    assert.equal(got.probe, true);
    assert.equal(got.forced, true);
  });

  it("probe throw publishes error + attempted artifact (not null / not a success)", () => {
    const script = join(tmpdir(), "milton-relaxed-kernel-probe-fail.mjs");
    writeFileSync(
      script,
      `import { RELAXED_DOT_PROBE } from ${JSON.stringify(join(ROOT, "src", "relaxed-simd.js"))};
const orig = WebAssembly.validate.bind(WebAssembly);
WebAssembly.validate = (buf) => {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.length === RELAXED_DOT_PROBE.length) {
    let same = true;
    for (let i = 0; i < u8.length; i += 1) {
      if (u8[i] !== RELAXED_DOT_PROBE[i]) { same = false; break; }
    }
    if (same) return false;
  }
  return orig(buf);
};
process.env.MILTON_RELAXED_SIMD = "1";
process.env.MILTON_THREADS = "1";
const m = await import(${JSON.stringify(INDEX)});
let threw = null;
try {
  await m.embed("hello", { prefix: "document" });
} catch (err) {
  threw = err instanceof Error ? err.message : String(err);
}
process.stdout.write(JSON.stringify({
  threw,
  kernel: m.lastQmatmulKernel,
  thread: m.lastThreadReport,
}));
`,
    );
    const ran = spawnSync(process.execPath, [script], {
      encoding: "utf8",
      timeout: 60000,
      env: { ...process.env, MILTON_THREADS: "1", MILTON_RELAXED_SIMD: "1" },
    });
    assert.equal(ran.status, 0, ran.stderr || ran.stdout);
    const got = JSON.parse(ran.stdout);
    assert.ok(got.threw && got.threw.includes("MILTON_RELAXED_SIMD=1"), got.threw);
    assert.equal(typeof got.kernel.error, "string");
    assert.ok(got.kernel.error.includes("MILTON_RELAXED_SIMD=1"), got.kernel.error);
    assert.equal(got.kernel.wasm, "milton_relaxed_bg.wasm");
    assert.equal(got.kernel.kernel, undefined);
    assert.equal(typeof got.thread.error, "string");
    assert.equal(got.thread.wasm, "milton_relaxed_bg.wasm");
    assert.equal(got.thread.artifact, "single");
  });
});
