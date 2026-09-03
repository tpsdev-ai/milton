#!/usr/bin/env node
/**
 * Two-way expected-outcome wrapper for `wasm:compare` (flair#1468 shape).
 *
 * compare-native-wasm.mjs still exits non-zero while native vs WASM disagree.
 * That non-zero is not the lane verdict: this file compares the receipt's
 * `result` to harness/expected.json. RED only when observed differs from
 * expected either way (unexpected PASS is red). An abort (no receipt) is
 * always RED — that is not a known residual.
 */
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const COMPARE = join(HERE, "compare-native-wasm.mjs");
const RECEIPT = join(ROOT, "harness", "receipts", "native-vs-wasm.json");
const DEFAULT_EXPECTED = join(ROOT, "harness", "expected.json");

/**
 * @param {string} raw
 * @param {string} [source]
 * @returns {{ expected: "pass" | "fail", reason: string, blocked_on?: number[] }}
 */
export function loadExpected(raw, source = "expected.json") {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`${source}: invalid JSON (${msg})`);
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error(`${source}: must be an object`);
  }
  if (parsed.expected !== "pass" && parsed.expected !== "fail") {
    throw new Error(`${source}: expected must be "pass" or "fail"`);
  }
  if (typeof parsed.reason !== "string" || parsed.reason.trim() === "") {
    throw new Error(`${source}: reason must document the live residual`);
  }
  return parsed;
}

/**
 * Tight-lane shape (Flint #53 ASK 2). Threaded verdicts must show
 * artifact=threads, wasm_threads=4, max_abs=0, and the expected kernel.
 * @param {object} receipt
 * @param {{ artifact?: "single" | "threads", threads?: number, kernel?: "relaxed" | "simd128", maxAbs?: number }} want
 * @returns {string[]}
 */
export function checkReceiptShape(receipt, want) {
  const misses = [];
  if (!receipt || typeof receipt !== "object") {
    return ["receipt is missing or not an object"];
  }
  if (want.artifact && receipt.wasm_artifact !== want.artifact) {
    misses.push(`wasm_artifact expected ${want.artifact} got ${JSON.stringify(receipt.wasm_artifact)}`);
  }
  if (want.threads != null && receipt.wasm_threads !== want.threads) {
    misses.push(`wasm_threads expected ${want.threads} got ${JSON.stringify(receipt.wasm_threads)}`);
  }
  if (want.kernel) {
    const got = receipt.qmatmul_kernel?.kernel;
    if (got !== want.kernel) {
      misses.push(`qmatmul_kernel.kernel expected ${want.kernel} got ${JSON.stringify(got)}`);
    }
  }
  if (want.maxAbs != null && receipt.max_abs !== want.maxAbs) {
    misses.push(`max_abs expected ${want.maxAbs} got ${JSON.stringify(receipt.max_abs)}`);
  }
  return misses;
}

/**
 * @param {{ expected: "pass" | "fail", observed: "pass" | "fail" | "abort", reason: string }} args
 * @returns {{ ok: boolean, summary: string }}
 */
export function judge({ expected, observed, reason }) {
  if (observed === "abort") {
    return {
      ok: false,
      summary:
        "RED: wasm:compare aborted (no receipt). That is not the known residual — fix the run.",
    };
  }
  if (observed === expected) {
    if (expected === "fail") {
      return {
        ok: true,
        summary: `GREEN: wasm:compare failed as expected.\n  ${reason}\n  Red means act on this PR: flip expected.json to pass when the residual is gone, or update reason if the residual changed.`,
      };
    }
    return {
      ok: true,
      summary: "GREEN: wasm:compare passed as expected.",
    };
  }
  if (observed === "pass" && expected === "fail") {
    return {
      ok: false,
      summary:
        "RED: unexpected PASS. Flip harness/expected.json to \"pass\" in this PR (or the compare went blind).",
    };
  }
  return {
    ok: false,
    summary: `RED: unexpected FAIL (expected pass).\n  Update harness/expected.json if this is a known residual, otherwise fix the regression.\n  last known reason: ${reason}`,
  };
}

function parseArgs(argv) {
  const out = {
    expected: DEFAULT_EXPECTED,
    help: false,
    skipRun: false,
    artifact: null,
    threads: null,
    kernel: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") out.help = true;
    else if (arg === "--expected") out.expected = argv[++i] ?? null;
    else if (arg === "--skip-run") out.skipRun = true;
    else if (arg === "--artifact") {
      const v = argv[++i];
      if (v !== "single" && v !== "threads") {
        throw new Error(`--artifact must be single or threads, got ${JSON.stringify(v)}`);
      }
      out.artifact = v;
    } else if (arg === "--threads") {
      const n = Number(argv[++i]);
      if (!Number.isInteger(n) || n < 1) {
        throw new Error(`--threads must be a positive integer, got ${JSON.stringify(argv[i])}`);
      }
      out.threads = n;
    } else if (arg === "--kernel") {
      const v = argv[++i];
      if (v !== "relaxed" && v !== "simd128") {
        throw new Error(`--kernel must be relaxed or simd128, got ${JSON.stringify(v)}`);
      }
      out.kernel = v;
    } else throw new Error(`unknown argument: ${arg}`);
  }
  return out;
}

/**
 * @param {string[]} argv
 * @returns {number}
 */
export function main(argv) {
  let args;
  try {
    args = parseArgs(argv);
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    return 2;
  }
  if (args.help) {
    process.stdout.write(
      "Usage: node harness/scripts/wasm-compare-verdict.mjs [--expected FILE] [--artifact single|threads] [--threads N] [--kernel relaxed|simd128]\n",
    );
    return 0;
  }
  if (!args.expected) {
    process.stderr.write("fail-closed: --expected PATH is required\n");
    return 2;
  }

  let expectedDoc;
  try {
    expectedDoc = loadExpected(readFileSync(args.expected, "utf8"), args.expected);
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    return 2;
  }

  if (!args.skipRun) {
    const env = { ...process.env };
    delete env.MILTON_ROPE_LIBM_SIN;
    delete env.MILTON_EMBED_BIN;
    // Single-thread lane sets MILTON_WASM_THREADS=0 in the npm script.
    // Threaded verdicts must not inherit that default — the product path
    // is auto threads on a 4-vCPU runner (Flint #53 ASK 2).
    if (args.artifact === "threads") {
      delete env.MILTON_WASM_THREADS;
      if (env.MILTON_THREADS === undefined || env.MILTON_THREADS === "") {
        env.MILTON_THREADS = args.threads != null ? String(args.threads) : "4";
      }
    } else if (env.MILTON_WASM_THREADS === undefined) {
      env.MILTON_WASM_THREADS = "0";
    }
    const ran = spawnSync(process.execPath, [COMPARE], {
      cwd: ROOT,
      encoding: "utf8",
      env,
      stdio: ["ignore", "pipe", "inherit"],
    });
    if (ran.stdout) process.stdout.write(ran.stdout);
    if (ran.error) {
      process.stderr.write(`fail-closed: spawn wasm:compare: ${ran.error.message}\n`);
      const verdict = judge({
        expected: expectedDoc.expected,
        observed: "abort",
        reason: expectedDoc.reason,
      });
      process.stdout.write(`${verdict.summary}\n`);
      return 1;
    }
  }

  let observed = "abort";
  let receipt = null;
  if (existsSync(RECEIPT)) {
    try {
      receipt = JSON.parse(readFileSync(RECEIPT, "utf8"));
      if (receipt.result === "pass" || receipt.result === "fail") {
        observed = receipt.result;
      }
    } catch {
      observed = "abort";
      receipt = null;
    }
  }

  const verdict = judge({
    expected: expectedDoc.expected,
    observed,
    reason: expectedDoc.reason,
  });
  const want = {
    artifact: args.artifact,
    threads: args.threads,
    kernel: args.kernel,
    maxAbs: args.artifact || args.kernel || args.threads != null ? 0 : null,
  };
  const shapeMisses = receipt
    ? checkReceiptShape(receipt, want)
    : args.artifact || args.kernel || args.threads != null
      ? ["receipt missing; cannot check artifact/threads/kernel/max_abs"]
      : [];

  process.stdout.write("==============================================\n");
  process.stdout.write(`${verdict.summary}\n`);
  process.stdout.write(`observed=${observed} expected=${expectedDoc.expected}\n`);
  if (receipt) {
    process.stdout.write(
      `wasm_artifact=${JSON.stringify(receipt.wasm_artifact)} wasm_threads=${JSON.stringify(receipt.wasm_threads)} max_abs=${JSON.stringify(receipt.max_abs)} qmatmul_kernel=${JSON.stringify(receipt.qmatmul_kernel)}\n`,
    );
  }
  if (shapeMisses.length) {
    process.stdout.write(`RED: receipt shape: ${shapeMisses.join("; ")}\n`);
  }
  process.stdout.write("==============================================\n");
  return verdict.ok && shapeMisses.length === 0 ? 0 : 1;
}

const isMain =
  process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));
if (isMain) {
  process.exit(main(process.argv.slice(2)));
}
