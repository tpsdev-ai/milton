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

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const INDEX = join(ROOT, "src", "index.js");

const CHILD = join(tmpdir(), "milton-relaxed-kernel-child.mjs");
writeFileSync(
  CHILD,
  `import { embed, lastQmatmulKernel } from ${JSON.stringify(INDEX)};
await embed("hello", { prefix: "document" });
process.stdout.write(JSON.stringify(lastQmatmulKernel));
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
    env: { ...process.env, MILTON_THREADS: "1", ...env },
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
    const got = run({});
    assert.deepEqual(Object.keys(got).sort(), ["forced", "kernel", "probe"]);
    assert.equal(got.kernel, "relaxed");
    assert.equal(got.probe, true);
    assert.equal(got.forced, false);
  });

  it("MILTON_RELAXED_SIMD=0 reports simd128 with probe still true", () => {
    const got = run({ MILTON_RELAXED_SIMD: "0" });
    assert.equal(got.kernel, "simd128");
    assert.equal(got.probe, true);
    assert.equal(got.forced, true);
  });

  it("MILTON_RELAXED_SIMD=1 reports relaxed forced", () => {
    const got = run({ MILTON_RELAXED_SIMD: "1" });
    assert.equal(got.kernel, "relaxed");
    assert.equal(got.probe, true);
    assert.equal(got.forced, true);
  });
});
