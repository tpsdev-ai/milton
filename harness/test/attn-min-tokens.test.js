import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ATTN_MIN_TOKENS_DEFAULT, ATTN_MIN_TOKENS_MAX } from "../../src/attn-min-tokens.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const SRC = join(ROOT, "src", "attn-min-tokens.js");

function clampRun(raw) {
  const ran = spawnSync(
    process.execPath,
    [
      "-e",
      `import { resolveAttnMinTokens, ATTN_MIN_TOKENS_DEFAULT } from ${JSON.stringify(SRC)};
const warnings = [];
console.warn = (msg) => { warnings.push(String(msg)); };
const raw = process.env.CLAMP_ATTN;
const env = raw === "__UNSET__" ? {} : { MILTON_ATTN_MIN_TOKENS: raw === "__EMPTY__" ? "" : raw };
const first = resolveAttnMinTokens(env);
const second = resolveAttnMinTokens(env);
process.stdout.write(JSON.stringify({ first, second, warnings, default: ATTN_MIN_TOKENS_DEFAULT }));`,
    ],
    {
      encoding: "utf8",
      timeout: 10000,
      env: { ...process.env, CLAMP_ATTN: raw },
    },
  );
  assert.equal(ran.status, 0, ran.stderr || ran.stdout);
  return JSON.parse(ran.stdout);
}

describe("MILTON_ATTN_MIN_TOKENS parse / clamp (#59)", () => {
  it("unset / empty use default 32 and do not warn", () => {
    for (const raw of ["__UNSET__", "__EMPTY__"]) {
      const got = clampRun(raw);
      assert.equal(got.first, ATTN_MIN_TOKENS_DEFAULT, raw);
      assert.equal(got.second, ATTN_MIN_TOKENS_DEFAULT, raw);
      assert.equal(got.warnings.length, 0, raw);
    }
  });

  it("in-range integers apply and do not warn", () => {
    for (const raw of ["1", "32", "64", String(ATTN_MIN_TOKENS_MAX), "3.9"]) {
      const got = clampRun(raw);
      const want = raw === "3.9" ? 3 : Number(raw);
      assert.equal(got.first, want, raw);
      assert.equal(got.second, want, raw);
      assert.equal(got.warnings.length, 0, raw);
    }
  });

  it("non-numeric / out-of-range fall back to 32 and warn once", () => {
    for (const raw of ["abc", "0", "-3", "NaN", "Infinity", "-Infinity", "99999"]) {
      const got = clampRun(raw);
      assert.equal(got.first, ATTN_MIN_TOKENS_DEFAULT, raw);
      assert.equal(got.second, ATTN_MIN_TOKENS_DEFAULT, raw);
      assert.equal(got.warnings.length, 1, raw);
      assert.ok(
        got.warnings[0].includes(`MILTON_ATTN_MIN_TOKENS=${raw}`),
        `expected warn naming MILTON_ATTN_MIN_TOKENS=${raw}, got ${JSON.stringify(got.warnings)}`,
      );
      assert.ok(
        got.warnings[0].includes(`using ${ATTN_MIN_TOKENS_DEFAULT}`),
        `expected warn naming applied ${ATTN_MIN_TOKENS_DEFAULT}, got ${JSON.stringify(got.warnings)}`,
      );
    }
  });
});
