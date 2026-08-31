#!/usr/bin/env node
/**
 * Generate pinned reference vectors + the CLS-pooled must-fail control.
 * Requires harness:setup (llama-embedding + verified GGUF).
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { corpusDigest, loadCorpus } from "../lib/corpus.js";
import { sha256file } from "../lib/digest.js";
import { loadPin, referenceDigest } from "../lib/goldens.js";
import { assertLlamaAtPin } from "../lib/llama-pin.js";
import { applyPrefix } from "../lib/prefix.js";
import { createReferenceEmbedder, readLlamaCppPin, resolvePaths, runLlamaEmbedding, PINNED_GGUF_SHA256 } from "../lib/reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, "..", "goldens");

const corpus = loadCorpus();
const paths = resolvePaths();
assertLlamaAtPin(paths.llamaDir, loadPin());
const embed = createReferenceEmbedder(paths);

const ggufSha = await sha256file(paths.gguf);
if (ggufSha !== PINNED_GGUF_SHA256) {
  throw new Error(`GGUF digest ${ggufSha} != pinned ${PINNED_GGUF_SHA256}`);
}

const llamaPin = readLlamaCppPin(paths.llamaDir);
const llamaCommit = execFileSync("git", ["-C", paths.llamaDir, "rev-parse", "HEAD"], {
  encoding: "utf8",
}).trim();
const llamaDigest = createHash("sha256").update(llamaCommit).digest("hex");
const llamaDescribe = execFileSync("git", ["-C", paths.llamaDir, "describe", "--always", "--tags"], {
  encoding: "utf8",
}).trim();

const items = [];
for (const c of corpus.cases) {
  process.stderr.write(`embed ${c.id} (${c.prefix}) ... `);
  try {
    const vector = await embed(c.text, { prefix: c.prefix });
    items.push({
      id: c.id,
      prefix: c.prefix,
      dims: vector.length,
      vector: Array.from(vector),
    });
    process.stderr.write(`ok dims=${vector.length}\n`);
  } catch (err) {
    process.stderr.write(`FAIL ${err.message}\n`);
    throw err;
  }
}

const controlId = "short-hello-document";
const controlCase = corpus.cases.find((c) => c.id === controlId);
process.stderr.write(`cls-pool control ${controlId} ... `);
const clsVec = await runLlamaEmbedding(applyPrefix(controlCase.text, controlCase.prefix), paths, [
  "--pooling",
  "cls",
]);
process.stderr.write(`ok dims=${clsVec.length}\n`);

const goldens = {
  schema: "milton.goldens/1",
  corpus_digest: corpusDigest(corpus),
  items,
};

const pin = {
  schema: "milton.pin/1",
  generated_at_utc: new Date().toISOString(),
  model: "nomic-embed-text-v1.5",
  gguf_file: "nomic-embed-text-v1.5.Q4_K_M.gguf",
  gguf_sha256: ggufSha,
  gguf_source: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
  pooling: "mean",
  embd_normalize: 2,
  threads: 1,
  ctx: 2048,
  batch: 2048,
  prefix_convention: {
    document: "search_document: {text}",
    query: "search_query: {text}",
    none: "{text}",
    source: "flair resources/embeddings-provider.ts + harper-fabric-embeddings src/engine.ts NOMIC_TEMPLATES",
  },
  llamacpp_commit: llamaCommit,
  llamacpp_digest: llamaDigest,
  llamacpp_describe: llamaDescribe,
  llamacpp_head_note: llamaPin,
  dims: items[0]?.dims ?? 768,
};

const controls = {
  schema: "milton.mustfail-controls/1",
  swap_pooling: {
    id: controlId,
    pooling: "cls",
    dims: clsVec.length,
    vector: Array.from(clsVec),
    note: "llama.cpp --pooling cls on the same prefixed text as short-hello-document. Must-fail control only.",
  },
};

mkdirSync(OUT_DIR, { recursive: true });
writeFileSync(join(OUT_DIR, "vectors.json"), `${JSON.stringify(goldens, null, 2)}\n`);
writeFileSync(join(OUT_DIR, "pin.json"), `${JSON.stringify(pin, null, 2)}\n`);
writeFileSync(join(OUT_DIR, "controls.json"), `${JSON.stringify(controls, null, 2)}\n`);

process.stdout.write(
  JSON.stringify(
    {
      n: items.length,
      corpus_digest: goldens.corpus_digest,
      reference_digest: referenceDigest(goldens),
      gguf_sha256: ggufSha,
      llamacpp_commit: llamaCommit,
      llamacpp_digest: llamaDigest,
    },
    null,
    2,
  ) + "\n",
);
