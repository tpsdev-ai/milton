#!/usr/bin/env node
/**
 * Run the golden-vector gate against an embedder.
 *
 *   node harness/scripts/run-gate.mjs              # replay (wiring check)
 *   node harness/scripts/run-gate.mjs --wrong perturb
 *   MILTON_GATE_EMBEDDER=milton node harness/scripts/run-gate.mjs
 *   MILTON_GATE_EMBEDDER=reference node harness/scripts/run-gate.mjs
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runGate } from "../lib/gate.js";
import { formatReceipt } from "../lib/receipts.js";
import { loadCorpus } from "../lib/corpus.js";
import { loadGoldens } from "../lib/goldens.js";
import { applyWrong, replayByCase } from "../lib/wrong-embedders.js";
import { createReferenceEmbedder } from "../lib/reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const wrongIdx = args.indexOf("--wrong");
const wrong = wrongIdx >= 0 ? args[wrongIdx + 1] : null;
const wantReference = args.includes("--reference") || process.env.MILTON_GATE_EMBEDDER === "reference";
const wantMilton = args.includes("--milton") || process.env.MILTON_GATE_EMBEDDER === "milton";

const corpus = loadCorpus();
const goldens = loadGoldens();

let embed;
let tag;
if (wrong) {
  const controls = JSON.parse(readFileSync(join(HERE, "..", "goldens", "controls.json"), "utf8"));
  embed = applyWrong(wrong, goldens, { corpus, control: controls.swap_pooling });
  tag = `wrong-${wrong}`;
} else if (wantMilton) {
  const milton = await import("../../src/index.js");
  embed = (text, opts) => milton.embed(text, opts);
  tag = "milton";
} else if (wantReference) {
  embed = createReferenceEmbedder();
  tag = "reference";
} else {
  embed = replayByCase(goldens, corpus);
  tag = "replay";
}

const receipt = await runGate(embed, { corpus, goldens });
const text = formatReceipt(receipt);
process.stdout.write(text + "\n");
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");

mkdirSync(join(HERE, "..", "receipts"), { recursive: true });
writeFileSync(join(HERE, "..", "receipts", `${tag}.json`), `${JSON.stringify(receipt, null, 2)}\n`);

if (receipt.result === "fail") process.exitCode = 1;
