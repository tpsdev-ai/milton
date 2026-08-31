import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { loadGoldens } from "../lib/goldens.js";
import { runGate } from "../lib/gate.js";
import { applyWrong, replayByCase } from "../lib/wrong-embedders.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const corpus = loadCorpus();
const goldens = loadGoldens();
const controls = JSON.parse(readFileSync(join(HERE, "..", "goldens", "controls.json"), "utf8"));

describe("must-fail control — gate rejects a deliberately-wrong embedder", () => {
  it("replay of pinned goldens is GREEN (wiring check, not a Milton embedder)", async () => {
    const receipt = await runGate(replayByCase(goldens, corpus), { corpus, goldens });
    assert.equal(receipt.result, "pass", JSON.stringify(receipt.failures, null, 2));
    assert.equal(receipt.failed, 0);
    assert.equal(receipt.n, corpus.cases.length);
  });

  it("perturb a vector turns the run RED and names failures", async () => {
    const receipt = await runGate(applyWrong("perturb", goldens, { corpus }), { corpus, goldens });
    assert.equal(receipt.result, "fail");
    assert.ok(receipt.failed > 0);
    assert.ok(receipt.failures.every((f) => f.id && f.delta));
    assert.ok(receipt.failures.some((f) => /max_abs|cos_dist/.test(f.delta.reason ?? "")));
  });

  it("drop the prefix turns the run RED and names the prefixed items", async () => {
    const receipt = await runGate(applyWrong("drop-prefix", goldens, { corpus }), { corpus, goldens });
    assert.equal(receipt.result, "fail");
    const ids = receipt.failures.map((f) => f.id);
    assert.ok(ids.includes("short-hello-document"), `named ${ids}`);
    assert.ok(ids.includes("short-hello-query"), `named ${ids}`);
    assert.ok(!ids.includes("short-hello-none"), "unprefixed sibling must still match");
  });

  it("swap pooling (CLS vs mean) turns the run RED and names the control item", async () => {
    const receipt = await runGate(
      applyWrong("swap-pooling", goldens, { corpus, control: controls.swap_pooling }),
      { corpus, goldens },
    );
    assert.equal(receipt.result, "fail");
    const ids = receipt.failures.map((f) => f.id);
    assert.ok(ids.includes("short-hello-document"), `named ${ids}`);
    assert.ok(ids.length >= 1);
  });

  it("truncate a dimension turns the run RED and names dim_mismatch", async () => {
    const receipt = await runGate(applyWrong("truncate", goldens, { corpus }), { corpus, goldens });
    assert.equal(receipt.result, "fail");
    assert.ok(receipt.failures.length > 0);
    assert.ok(receipt.failures.every((f) => f.delta.reason?.startsWith("dim_mismatch")));
  });
});
