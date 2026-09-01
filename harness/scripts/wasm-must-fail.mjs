#!/usr/bin/env node
/**
 * Must-fail on the WASM path: a deliberately-wrong embedder must turn RED.
 * Uses the same Rust ForwardFault as embed-must-fail (not a JS-only stub).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { loadGoldens } from "../lib/goldens.js";
import { runGate } from "../lib/gate.js";
import { embed } from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const corpus = loadCorpus();
const goldens = loadGoldens();

const faults = ["layernorm", "pooling", "dropped-prefix"];
const named = [];
const slipped = [];

for (const fault of faults) {
  const receipt = await runGate((text, opts) => embed(text, { prefix: opts.prefix, fault }), {
    corpus,
    goldens,
  });
  const line = `${fault} ${receipt.result} ${receipt.failed} cases`;
  named.push(line);
  if (receipt.result !== "fail" || receipt.failed === 0) {
    slipped.push(line);
  }
}

const out = {
  schema: "milton.wasm.must-fail/1",
  result: slipped.length === 0 ? "pass" : "fail",
  named,
  slipped,
};
mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(join(ROOT, "harness", "receipts", "wasm-must-fail.json"), `${JSON.stringify(out, null, 2)}\n`);
process.stdout.write(JSON.stringify(out, null, 2) + "\n");
if (out.result === "fail") process.exitCode = 1;
