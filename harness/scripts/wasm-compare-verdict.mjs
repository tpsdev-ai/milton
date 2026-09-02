#!/usr/bin/env node
/**
 * Two-way expected-outcome wrapper for `wasm:compare` (flair#1468 shape).
 *
 * compare-native-wasm.mjs still exits non-zero while native vs WASM disagree.
 * That non-zero is not the lane verdict: this file compares the receipt's
 * `result` to harness/expected.json. RED only when observed differs from
 * expected either way (unexpected PASS is red). An abort (no receipt) is
 * always RED — that is not the known softmax residual.
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
  const out = { expected: DEFAULT_EXPECTED, help: false, skipRun: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") out.help = true;
    else if (arg === "--expected") out.expected = argv[++i] ?? null;
    else if (arg === "--skip-run") out.skipRun = true;
    else throw new Error(`unknown argument: ${arg}`);
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
      "Usage: node harness/scripts/wasm-compare-verdict.mjs [--expected FILE]\n",
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
    const ran = spawnSync(process.execPath, [COMPARE], {
      cwd: ROOT,
      encoding: "utf8",
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
  if (existsSync(RECEIPT)) {
    try {
      const receipt = JSON.parse(readFileSync(RECEIPT, "utf8"));
      if (receipt.result === "pass" || receipt.result === "fail") {
        observed = receipt.result;
      }
    } catch {
      observed = "abort";
    }
  }

  const verdict = judge({
    expected: expectedDoc.expected,
    observed,
    reason: expectedDoc.reason,
  });
  process.stdout.write("==============================================\n");
  process.stdout.write(`${verdict.summary}\n`);
  process.stdout.write(`observed=${observed} expected=${expectedDoc.expected}\n`);
  process.stdout.write("==============================================\n");
  return verdict.ok ? 0 : 1;
}

const isMain =
  process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));
if (isMain) {
  process.exit(main(process.argv.slice(2)));
}
