#!/usr/bin/env node
/**
 * Generate pinned reference vectors + the CLS-pooled must-fail control.
 * Requires harness:setup (llama-embedding + verified GGUF).
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { corpusDigest, loadCorpus } from "../lib/corpus.js";
import { sha256file } from "../lib/digest.js";
import { loadPin, referenceDigest } from "../lib/goldens.js";
import { assertLlamaAtPin } from "../lib/llama-pin.js";
import { applyPrefix } from "../lib/prefix.js";
import {
  createReferenceEmbedder,
  readLlamaCppPin,
  resolvePaths,
  runEmbedFromTokenIds,
  runLlamaEmbedding,
  PINNED_GGUF_SHA256,
} from "../lib/reference.js";
import { TOKEN_ID_ORACLE_CASES, TOKEN_ID_ORACLE_SOURCE, TOKEN_ID_ORACLE_TOOL } from "../lib/token-id-oracle.js";

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

const onlyOracle = process.argv.includes("--only-token-id-oracle");
const tokensPath = join(HERE, "..", "goldens", "tokens.json");
const tokens = JSON.parse(readFileSync(tokensPath, "utf8"));
const tokensById = new Map((tokens.items || []).map((it) => [it.id, it]));

const existingById = new Map();
if (onlyOracle) {
  const existingPath = join(OUT_DIR, "vectors.json");
  const existing = JSON.parse(readFileSync(existingPath, "utf8"));
  for (const it of existing.items) existingById.set(it.id, it);
}

const items = [];
for (const c of corpus.cases) {
  const useTokenIds = TOKEN_ID_ORACLE_CASES.includes(c.id);
  if (onlyOracle && !useTokenIds) {
    const prev = existingById.get(c.id);
    if (!prev) throw new Error(`fail-closed: --only-token-id-oracle missing existing golden ${c.id}`);
    items.push(prev);
    continue;
  }
  process.stderr.write(`embed ${c.id} (${c.prefix})${useTokenIds ? " [token-ids]" : ""} ... `);
  try {
    if (useTokenIds) {
      const tok = tokensById.get(c.id);
      if (!tok?.ids?.length) {
        throw new Error(`missing HF token IDs in ${TOKEN_ID_ORACLE_SOURCE} for ${c.id}`);
      }
      const { vector, provenance } = runEmbedFromTokenIds(tok.ids, paths);
      items.push({
        id: c.id,
        prefix: c.prefix,
        dims: vector.length,
        vector: Array.from(vector),
        provenance,
      });
    } else {
      const vector = await embed(c.text, { prefix: c.prefix });
      items.push({
        id: c.id,
        prefix: c.prefix,
        dims: vector.length,
        vector: Array.from(vector),
      });
    }
    process.stderr.write(`ok dims=${items[items.length - 1].dims}\n`);
  } catch (err) {
    process.stderr.write(`FAIL ${err.message}\n`);
    throw err;
  }
}

const controlId = "short-hello-document";
const controlCase = corpus.cases.find((c) => c.id === controlId);
let clsVec = null;
if (!onlyOracle) {
  process.stderr.write(`cls-pool control ${controlId} ... `);
  clsVec = await runLlamaEmbedding(applyPrefix(controlCase.text, controlCase.prefix), paths, [
    "--pooling",
    "cls",
  ]);
  process.stderr.write(`ok dims=${clsVec.length}\n`);
}

const goldens = {
  schema: "milton.goldens/1",
  corpus_digest: corpusDigest(corpus),
  token_id_oracle: {
    tool: TOKEN_ID_ORACLE_TOOL,
    token_ids_source: TOKEN_ID_ORACLE_SOURCE,
    cases: [...TOKEN_ID_ORACLE_CASES],
    note: "HF token IDs (tokens.json) through llama.cpp GGUF forward via embed-from-token-ids. Not the llama-embedding text path.",
  },
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

const controls = onlyOracle
  ? null
  : {
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
if (!onlyOracle) {
  writeFileSync(join(OUT_DIR, "pin.json"), `${JSON.stringify(pin, null, 2)}\n`);
  writeFileSync(join(OUT_DIR, "controls.json"), `${JSON.stringify(controls, null, 2)}\n`);
}

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
