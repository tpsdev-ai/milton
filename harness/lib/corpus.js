import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { sha256json } from "./digest.js";
import { PREFIX_KINDS } from "./prefix.js";

const HERE = dirname(fileURLToPath(import.meta.url));
export const CORPUS_PATH = join(HERE, "..", "corpus", "corpus.json");

export function loadCorpus(path = CORPUS_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (!raw || raw.schema !== "milton.corpus/1" || !Array.isArray(raw.cases)) {
    throw new Error(`invalid corpus at ${path}: expected schema milton.corpus/1`);
  }
  const ids = new Set();
  for (const c of raw.cases) {
    if (!c.id || typeof c.text !== "string" || !PREFIX_KINDS.includes(c.prefix)) {
      throw new Error(`invalid corpus case: ${JSON.stringify({ id: c.id, prefix: c.prefix })}`);
    }
    if (!c.trap || typeof c.trap !== "string") {
      throw new Error(`corpus case ${c.id} is missing a documented trap (failure mode)`);
    }
    if (ids.has(c.id)) throw new Error(`duplicate corpus id: ${c.id}`);
    ids.add(c.id);
  }
  return raw;
}

/** Digest over {id, text, prefix} only — trap prose can be edited without invalidating goldens. */
export function corpusDigest(corpus) {
  return sha256json(
    corpus.cases.map((c) => ({ id: c.id, text: c.text, prefix: c.prefix })),
  );
}
