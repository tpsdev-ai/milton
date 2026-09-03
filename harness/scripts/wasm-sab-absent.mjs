#!/usr/bin/env node
/**
 * SAB-absent loader lane (issue #44).
 *
 * Deletes SharedArrayBuffer (and optionally Atomics) before importing the
 * public API. The loader must pick wasm/milton_bg.wasm and embed without
 * treating the absence as an error. No fallback avalanche.
 *
 * Node 22 has no --no-harmony-sharedarraybuffer. This is the simulation
 * the CI lane runs. `--experimental-wasm-threads` is already default-on
 * here; the probe is the capability check, not a Node flag.
 */
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const CHILD = join(HERE, "wasm-sab-absent-child.mjs");

const ran = spawnSync(process.execPath, [CHILD], {
  cwd: ROOT,
  encoding: "utf8",
  env: { ...process.env, MILTON_WASM_THREADS: "1" },
});
if (ran.stdout) process.stdout.write(ran.stdout);
if (ran.stderr) process.stderr.write(ran.stderr);
if (ran.error) {
  process.stderr.write(`fail-closed: spawn sab-absent child: ${ran.error.message}\n`);
  process.exit(2);
}
process.exit(ran.status === 0 ? 0 : ran.status ?? 1);
