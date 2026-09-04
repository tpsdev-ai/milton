import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ATTN_MIN_TOKENS_DEFAULT, resolveAttnMinTokens } from "../../src/attn-min-tokens.js";
import {
  ATTN_CROSSOVER_PINS,
  COMPARE_CROSSOVER_PATH,
  corpusDigest,
  loadCompareCorpus,
  loadCorpus,
  loadCrossoverCases,
} from "../lib/corpus.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const TOKENS_PATH = join(HERE, "..", "goldens", "tokens.json");

describe("ATTN_PARALLEL_MIN_TOKENS=32 compare crossover (#58)", () => {
  const conformance = loadCorpus();
  const compare = loadCompareCorpus();
  const extra = loadCrossoverCases();
  const gold = JSON.parse(readFileSync(TOKENS_PATH, "utf8"));

  it("does not touch the 18-case conformance corpus or goldens digest", () => {
    assert.equal(conformance.cases.length, 18);
    assert.equal(gold.n, 18);
    assert.equal(corpusDigest(conformance), gold.corpus_digest);
    const compareIds = new Set(compare.cases.map((c) => c.id));
    for (const c of conformance.cases) {
      assert.ok(compareIds.has(c.id), `compare corpus dropped ${c.id}`);
    }
  });

  it("pins n=31/32/33 on compare-only hello-repeat fixtures", () => {
    assert.equal(extra.gate, 32);
    assert.equal(extra.cases.length, 3);
    assert.deepEqual(
      extra.cases.map((c) => ({ id: c.id, n_tokens: c.n_tokens, prefix: c.prefix })),
      ATTN_CROSSOVER_PINS.map((p) => ({ ...p, prefix: "document" })),
    );
    assert.equal(compare.cases.length, 21);
    assert.equal(compare.crossover.length, 3);
    const hellos = { 31: 25, 32: 26, 33: 27 };
    for (const pin of ATTN_CROSSOVER_PINS) {
      const c = extra.cases.find((x) => x.id === pin.id);
      assert.ok(c, pin.id);
      assert.equal(c.text.split(" ").length, hellos[pin.n_tokens], pin.id);
      assert.ok(c.text.split(" ").every((w) => w === "hello"), pin.id);
      assert.match(c.trap, /ATTN_PARALLEL_MIN_TOKENS|crossover|gate/i);
    }
  });

  it("crossover JSON path is the committed compare-only fixture", () => {
    assert.ok(COMPARE_CROSSOVER_PATH.endsWith("harness/corpus/compare-crossover.json"));
    const raw = JSON.parse(readFileSync(COMPARE_CROSSOVER_PATH, "utf8"));
    assert.equal(raw.schema, "milton.compare-crossover/1");
    assert.equal(raw.gate, 32);
  });

  it("JS default and wasm getter equal fixture gate: 32 (Flint rider)", () => {
    assert.equal(extra.gate, ATTN_MIN_TOKENS_DEFAULT);
    assert.equal(resolveAttnMinTokens({}), extra.gate);
    assert.equal(resolveAttnMinTokens({ MILTON_ATTN_MIN_TOKENS: "" }), extra.gate);
    const got = runAttnGate({});
    assert.equal(got.resolved, extra.gate);
    assert.equal(got.applied, extra.gate);
    assert.equal(got.wasm, extra.gate);
  });

  it("MILTON_ATTN_MIN_TOKENS overrides the effective wasm gate", () => {
    const got = runAttnGate({ MILTON_ATTN_MIN_TOKENS: "64" });
    assert.equal(got.resolved, 64);
    assert.equal(got.applied, 64);
    assert.equal(got.wasm, 64);
  });
});

function runAttnGate(env) {
  const ROOT = join(HERE, "../..");
  const SRC = join(ROOT, "src", "attn-min-tokens.js");
  const GLUE = join(ROOT, "wasm", "milton_relaxed.js");
  const WASM = join(ROOT, "wasm", "milton_relaxed_bg.wasm");
  const ran = spawnSync(
    process.execPath,
    [
      "-e",
      `import { readFileSync } from "node:fs";
import { applyAttnMinTokens, resolveAttnMinTokens } from ${JSON.stringify(SRC)};
import init, { attnMinTokens, attnSetMinTokens } from ${JSON.stringify(GLUE)};
await init({ module_or_path: readFileSync(${JSON.stringify(WASM)}) });
const applied = applyAttnMinTokens({ attnSetMinTokens, attnMinTokens });
process.stdout.write(JSON.stringify({
  resolved: resolveAttnMinTokens(),
  applied,
  wasm: attnMinTokens(),
}));`,
    ],
    {
      encoding: "utf8",
      timeout: 30000,
      env: { ...process.env, ...env },
    },
  );
  assert.equal(ran.status, 0, ran.stderr || ran.stdout);
  return JSON.parse(ran.stdout);
}
