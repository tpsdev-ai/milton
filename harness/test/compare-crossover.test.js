import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
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
});
