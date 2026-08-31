#!/usr/bin/env node
/**
 * Check out (or assert) pin.json's llama.cpp commit. Fail closed on drift.
 *
 *   node harness/scripts/ensure-llama-pin.mjs --print
 *   node harness/scripts/ensure-llama-pin.mjs --checkout --dir <llamaDir>
 *   node harness/scripts/ensure-llama-pin.mjs --assert --dir <llamaDir>
 */
import { loadPin } from "../lib/goldens.js";
import { resolvePaths } from "../lib/reference.js";
import { assertLlamaAtPin, checkoutLlamaPin, pinnedLlamaCommit } from "../lib/llama-pin.js";

const args = process.argv.slice(2);
const pin = loadPin();
const dirIdx = args.indexOf("--dir");
const llamaDir = dirIdx >= 0 ? args[dirIdx + 1] : resolvePaths().llamaDir;

if (args.includes("--print")) {
  process.stdout.write(`${pinnedLlamaCommit(pin)}\n`);
  process.exit(0);
}

if (args.includes("--checkout")) {
  const head = checkoutLlamaPin(llamaDir, pin, { fetch: true });
  process.stdout.write(`llama.cpp HEAD ${head} matches pin.json\n`);
  process.exit(0);
}

assertLlamaAtPin(llamaDir, pin);
process.stdout.write(`llama.cpp HEAD ${pinnedLlamaCommit(pin)} matches pin.json\n`);
