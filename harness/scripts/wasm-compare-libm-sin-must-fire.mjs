#!/usr/bin/env node
/**
 * Must-fire wrapper: native libm sin vs WASM shared sin must turn
 * wasm:compare RED. Builds milton-embed with --features rope-libm-sin
 * (compare-native-wasm.mjs rebuilds when MILTON_ROPE_LIBM_SIN=1).
 * A pass here is a dead lock — fail-closed.
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const COMPARE = join(HERE, "compare-native-wasm.mjs");
const RECEIPT = join(ROOT, "harness", "receipts", "native-vs-wasm.json");
const MUST_FIRE = join(ROOT, "harness", "receipts", "native-vs-wasm-libm-sin.json");

const ran = spawnSync(process.execPath, [COMPARE], {
  cwd: ROOT,
  encoding: "utf8",
  env: { ...process.env, MILTON_ROPE_LIBM_SIN: "1" },
  stdio: ["ignore", "pipe", "inherit"],
});
if (ran.stdout) process.stdout.write(ran.stdout);
if (ran.error) {
  process.stderr.write(`fail-closed: spawn wasm:compare: ${ran.error.message}\n`);
  process.exit(1);
}

if (!existsSync(RECEIPT)) {
  process.stderr.write("fail-closed: must-fire produced no native-vs-wasm receipt\n");
  process.exit(1);
}

let receipt;
try {
  receipt = JSON.parse(readFileSync(RECEIPT, "utf8"));
} catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
  process.stderr.write(`fail-closed: must-fire receipt is not JSON (${msg})\n`);
  process.exit(1);
}

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(MUST_FIRE, `${JSON.stringify(receipt, null, 2)}\n`);

const n = receipt.n;
const failed = receipt.failed;
const result = receipt.result;
const maxCos = receipt.max_cos_dist;
const maxAbs = receipt.max_abs;

process.stdout.write("==============================================\n");
if (result === "fail" && Number(failed) > 0 && ran.status !== 0) {
  process.stdout.write(
    `GREEN: must-fire RED as required.\n  result=${result} n=${n} failed=${failed} max_cos_dist=${maxCos} max_abs=${maxAbs}\n`,
  );
  process.stdout.write("==============================================\n");
  process.exit(0);
}

process.stderr.write(
  `fail-closed: must-fire did not turn wasm:compare RED (result=${result} failed=${failed} status=${ran.status}). A gate nobody has seen fail is not a gate.\n`,
);
process.stdout.write("==============================================\n");
process.exit(1);
