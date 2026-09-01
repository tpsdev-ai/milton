#!/usr/bin/env node
/**
 * Generate F16 (MOSTLY_F16) llama-embedding vectors for the conformance corpus.
 * These are the F32-math ground-truth oracle (original HF F16 weights, F32 compute).
 * The GGUF itself is gitignored; this script writes embeddings + pin only.
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { corpusDigest, loadCorpus } from "../lib/corpus.js";
import { sha256file } from "../lib/digest.js";
import { loadEpsilon } from "../lib/goldens.js";
import { compareVectors } from "../lib/metrics.js";
import {
  createReferenceEmbedder,
  readLlamaCppPin,
  resolvePaths,
  runEmbedFromTokenIds,
} from "../lib/reference.js";
import { TOKEN_ID_ORACLE_CASES, TOKEN_ID_ORACLE_SOURCE, TOKEN_ID_ORACLE_TOOL } from "../lib/token-id-oracle.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const OUT_DIR = join(HERE, "..", "goldens");
const F16_GGUF = join(ROOT, "harness/vendor/models/nomic-embed-text-v1.5.F16.gguf");

const corpus = loadCorpus();
const paths = resolvePaths();
paths.gguf = process.env.MILTON_F16_GGUF || F16_GGUF;
const embed = createReferenceEmbedder(paths);

const ggufSha = await sha256file(paths.gguf);
const pinF16 = JSON.parse(readFileSync(join(OUT_DIR, "pin-f16.json"), "utf8"));
const onlyOracle = process.argv.includes("--only-token-id-oracle");
let f16FileNote = null;
if (ggufSha !== pinF16.gguf_sha256) {
  if (!onlyOracle) {
    throw new Error(
      `fail-closed: F16 GGUF digest ${ggufSha} != pin-f16.json ${pinF16.gguf_sha256}`,
    );
  }
  // Same nomic F16 weights can be packaged with a different GGUF sha
  // (official nomic-embed-text-v1.5.f16.gguf vs convert_hf_to_gguf.py).
  // Fail-closed: the file must reproduce the committed unicode-nfc F16 golden
  // from the same HF token IDs before we trust it for the #15 cases.
  const tokensForCheck = JSON.parse(readFileSync(join(OUT_DIR, "tokens.json"), "utf8"));
  const nfcTok = (tokensForCheck.items || []).find((it) => it.id === "unicode-nfc");
  const existingF16 = JSON.parse(readFileSync(join(OUT_DIR, "vectors-f16.json"), "utf8"));
  const nfcGolden = (existingF16.items || []).find((it) => it.id === "unicode-nfc");
  if (!nfcTok?.ids?.length || !nfcGolden?.vector) {
    throw new Error("fail-closed: cannot prove F16 file equivalence (missing unicode-nfc)");
  }
  const { vector: nfcGot } = runEmbedFromTokenIds(nfcTok.ids, paths);
  const eps = loadEpsilon();
  const cmp = compareVectors(nfcGot, nfcGolden.vector, {
    epsilon: eps.epsilon,
    epsilonAbs: eps.epsilon_abs,
  });
  if (!cmp.pass) {
    throw new Error(
      `fail-closed: F16 GGUF ${ggufSha} is not vector-equivalent to pin-f16 ${pinF16.gguf_sha256} ` +
        `(unicode-nfc cos_dist=${cmp.cos_dist} max_abs=${cmp.max_abs})`,
    );
  }
  f16FileNote = {
    used_gguf_sha256: ggufSha,
    pin_f16_sha256: pinF16.gguf_sha256,
    equivalence: "unicode-nfc token-id embed matches committed F16 golden",
    cos_dist: cmp.cos_dist,
    max_abs: cmp.max_abs,
  };
  process.stderr.write(
    `F16 GGUF sha differs from pin-f16; unicode-nfc equivalence ok cos_dist=${cmp.cos_dist} max_abs=${cmp.max_abs}\n`,
  );
}
const llamaPin = readLlamaCppPin(paths.llamaDir);

const tokensPath = join(HERE, "..", "goldens", "tokens.json");
const tokens = JSON.parse(readFileSync(tokensPath, "utf8"));
const tokensById = new Map((tokens.items || []).map((it) => [it.id, it]));

const existingById = new Map();
if (onlyOracle) {
  const existing = JSON.parse(readFileSync(join(OUT_DIR, "vectors-f16.json"), "utf8"));
  for (const it of existing.items) existingById.set(it.id, it);
}

const items = [];
for (const c of corpus.cases) {
  const useTokenIds = TOKEN_ID_ORACLE_CASES.includes(c.id);
  if (onlyOracle && !useTokenIds) {
    const prev = existingById.get(c.id);
    if (!prev) throw new Error(`fail-closed: --only-token-id-oracle missing existing F16 golden ${c.id}`);
    items.push(prev);
    continue;
  }
  process.stderr.write(`f16 ${c.id} (${c.prefix})${useTokenIds ? " [token-ids]" : ""} ... `);
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
      provenance: f16FileNote ? { ...provenance, f16_file: f16FileNote } : provenance,
    });
    process.stderr.write(`ok dims=${vector.length} d0=${vector[0].toFixed(8)}\n`);
  } else {
    const vector = await embed(c.text, { prefix: c.prefix });
    items.push({
      id: c.id,
      prefix: c.prefix,
      dims: vector.length,
      vector: Array.from(vector),
    });
    process.stderr.write(`ok dims=${vector.length} d0=${vector[0].toFixed(8)}\n`);
  }
}

const goldens = {
  schema: "milton.goldens.f16/1",
  note: "llama-embedding on original nomic-embed-text-v1.5 F16 GGUF (HF → convert_hf_to_gguf.py --outtype f16). F32-math oracle. NOT a dequantized Q4_K_M. unicode-nfd / newlines-tabs use embed-from-token-ids on the same GGUF with HF token IDs from tokens.json.",
  corpus_digest: corpusDigest(corpus),
  token_id_oracle: {
    tool: TOKEN_ID_ORACLE_TOOL,
    token_ids_source: TOKEN_ID_ORACLE_SOURCE,
    cases: [...TOKEN_ID_ORACLE_CASES],
    note: "HF token IDs through llama.cpp GGUF forward. Same F16 weights as the other 16 cases.",
    ...(f16FileNote ? { f16_file: f16FileNote } : {}),
  },
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
if (!onlyOracle) {
  writeFileSync(join(OUT_DIR, "pin-f16.json"), `${JSON.stringify(pin, null, 2)}\n`);
}
process.stderr.write(`wrote vectors-f16.json n=${items.length} sha256=${ggufSha}\n`);
