import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { sha256json } from "./digest.js";
import { PREFIX_KINDS } from "./prefix.js";

const HERE = dirname(fileURLToPath(import.meta.url));
export const CORPUS_PATH = join(HERE, "..", "corpus", "corpus.json");
export const COMPARE_CROSSOVER_PATH = join(HERE, "..", "corpus", "compare-crossover.json");

/** ATTN_PARALLEL_MIN_TOKENS=32 serial→parallel pins (compare-only). */
export const ATTN_CROSSOVER_PINS = Object.freeze([
  Object.freeze({ id: "attn-crossover-31", n_tokens: 31 }),
  Object.freeze({ id: "attn-crossover-32", n_tokens: 32 }),
  Object.freeze({ id: "attn-crossover-33", n_tokens: 33 }),
]);

function validateCase(c, extras = {}) {
  if (!c.id || typeof c.text !== "string" || !PREFIX_KINDS.includes(c.prefix)) {
    throw new Error(`invalid corpus case: ${JSON.stringify({ id: c.id, prefix: c.prefix })}`);
  }
  if (!c.trap || typeof c.trap !== "string") {
    throw new Error(`corpus case ${c.id} is missing a documented trap (failure mode)`);
  }
  if (extras.n_tokens != null) {
    if (!Number.isInteger(c.n_tokens) || c.n_tokens !== extras.n_tokens) {
      throw new Error(
        `corpus case ${c.id}: n_tokens expected ${extras.n_tokens} got ${JSON.stringify(c.n_tokens)}`,
      );
    }
  }
}

export function loadCorpus(path = CORPUS_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (!raw || raw.schema !== "milton.corpus/1" || !Array.isArray(raw.cases)) {
    throw new Error(`invalid corpus at ${path}: expected schema milton.corpus/1`);
  }
  const ids = new Set();
  for (const c of raw.cases) {
    validateCase(c);
    if (ids.has(c.id)) throw new Error(`duplicate corpus id: ${c.id}`);
    ids.add(c.id);
  }
  return raw;
}

/** Compare-only crossover fixtures. Does not change the 18-case goldens digest. */
export function loadCrossoverCases(path = COMPARE_CROSSOVER_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (!raw || raw.schema !== "milton.compare-crossover/1" || !Array.isArray(raw.cases)) {
    throw new Error(`invalid crossover corpus at ${path}: expected schema milton.compare-crossover/1`);
  }
  if (raw.gate !== 32) {
    throw new Error(`crossover corpus gate expected 32 got ${JSON.stringify(raw.gate)}`);
  }
  const byId = new Map(raw.cases.map((c) => [c.id, c]));
  const ids = new Set();
  for (const pin of ATTN_CROSSOVER_PINS) {
    const c = byId.get(pin.id);
    if (!c) throw new Error(`crossover corpus missing pinned case ${pin.id}`);
    validateCase(c, pin);
    if (ids.has(c.id)) throw new Error(`duplicate crossover id: ${c.id}`);
    ids.add(c.id);
  }
  if (raw.cases.length !== ATTN_CROSSOVER_PINS.length) {
    throw new Error(
      `crossover corpus: expected ${ATTN_CROSSOVER_PINS.length} cases got ${raw.cases.length}`,
    );
  }
  return raw;
}

/**
 * Corpus used by wasm:compare / threaded product-path compare:
 * 18 conformance cases plus the ATTN_PARALLEL_MIN_TOKENS=32 pins.
 * Goldens / wasm:gate still use loadCorpus() only.
 */
export function loadCompareCorpus(
  conformancePath = CORPUS_PATH,
  crossoverPath = COMPARE_CROSSOVER_PATH,
) {
  const base = loadCorpus(conformancePath);
  const extra = loadCrossoverCases(crossoverPath);
  const ids = new Set(base.cases.map((c) => c.id));
  for (const c of extra.cases) {
    if (ids.has(c.id)) throw new Error(`duplicate compare case id: ${c.id}`);
    ids.add(c.id);
  }
  return {
    schema: base.schema,
    model: base.model,
    notes: "conformance corpus + ATTN_PARALLEL_MIN_TOKENS=32 crossover (compare-only)",
    cases: [...base.cases, ...extra.cases],
    crossover: extra.cases,
    gate: extra.gate,
  };
}

/** Digest over {id, text, prefix} only — trap prose can be edited without invalidating goldens. */
export function corpusDigest(corpus) {
  return sha256json(
    corpus.cases.map((c) => ({ id: c.id, text: c.text, prefix: c.prefix })),
  );
}
