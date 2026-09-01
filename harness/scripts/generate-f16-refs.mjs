#!/usr/bin/env node
/**
 * Generate F16 (MOSTLY_F16) llama-embedding vectors for the conformance corpus.
 * These are the F32-math ground-truth oracle (original HF F16 weights, F32 compute).
 * The GGUF itself is gitignored; this script writes embeddings + pin only.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { corpusDigest, loadCorpus } from "../lib/corpus.js";
import { sha256file } from "../lib/digest.js";
import { applyPrefix } from "../lib/prefix.js";
import {
  createReferenceEmbedder,
  readLlamaCppPin,
  resolvePaths,
  runLlamaEmbedding,
} from "../lib/reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const OUT_DIR = join(HERE, "..", "goldens");
const F16_GGUF = join(ROOT, "harness/vendor/models/nomic-embed-text-v1.5.F16.gguf");

const corpus = loadCorpus();
const paths = resolvePaths();
paths.gguf = process.env.MILTON_F16_GGUF || F16_GGUF;
const embed = createReferenceEmbedder(paths);

const ggufSha = await sha256file(paths.gguf);
const llamaPin = readLlamaCppPin(paths.llamaDir);

const items = [];
for (const c of corpus.cases) {
  process.stderr.write(`f16 ${c.id} (${c.prefix}) ... `);
  const vector = await embed(c.text, { prefix: c.prefix });
  items.push({
    id: c.id,
    prefix: c.prefix,
    dims: vector.length,
    vector: Array.from(vector),
  });
  process.stderr.write(`ok dims=${vector.length} d0=${vector[0].toFixed(8)}\n`);
}

const goldens = {
  schema: "milton.goldens.f16/1",
  note: "llama-embedding on original nomic-embed-text-v1.5 F16 GGUF (HF → convert_hf_to_gguf.py --outtype f16). F32-math oracle. NOT a dequantized Q4_K_M.",
  corpus_digest: corpusDigest(corpus),
  items,
};

const pin = {
  schema: "milton.pin.f16/1",
  generated_at_utc: new Date().toISOString(),
  model: "nomic-embed-text-v1.5",
  gguf_file: "nomic-embed-text-v1.5.F16.gguf",
  gguf_sha256: ggufSha,
  gguf_file_type: "MOSTLY_F16",
  gguf_source:
    "nomic-ai/nomic-embed-text-v1.5 via convert_hf_to_gguf.py --outtype f16 (NOT dequantized Q4_K_M)",
  pooling: "mean",
  embd_normalize: 2,
  threads: 1,
  ctx: 2048,
  batch: 2048,
  llamacpp_commit: llamaPin.commit,
  llamacpp_describe: llamaPin.describe,
  dims: items[0]?.dims ?? 768,
};

mkdirSync(OUT_DIR, { recursive: true });
writeFileSync(join(OUT_DIR, "vectors-f16.json"), `${JSON.stringify(goldens, null, 2)}\n`);
writeFileSync(join(OUT_DIR, "pin-f16.json"), `${JSON.stringify(pin, null, 2)}\n`);
process.stderr.write(`wrote vectors-f16.json n=${items.length} sha256=${ggufSha}\n`);
