/**
 * Deliberately-wrong embedders. The gate is not trusted until each of these
 * turns a run RED and names the failure (HARNESS-SPEC.md).
 *
 * These are NOT Milton. They exist only to prove the gate rejects.
 */

import { asFloat32, goldensById } from "./goldens.js";

function caseKey(text, prefix) {
  return `${prefix}\0${text}`;
}

export function indexByCase(goldens, corpus) {
  const byId = goldensById(goldens);
  const map = new Map();
  for (const c of corpus.cases) {
    const item = byId.get(c.id);
    if (item) map.set(caseKey(c.text, c.prefix), { case: c, item });
  }
  return map;
}

/** Replay the pinned golden — the only "correct" path in this harness-only PR. */
export function replayByCase(goldens, corpus) {
  const map = indexByCase(goldens, corpus);
  return async function embed(text, { prefix } = {}) {
    const hit = map.get(caseKey(text, prefix));
    if (!hit) throw new Error(`replayByCase: no golden for prefix=${prefix}`);
    return asFloat32(hit.item.vector);
  };
}

/** 1. Perturb a vector — shift one dimension. */
export function perturbEmbedder(goldens, corpus, { dim = 0, delta = 1 } = {}) {
  const inner = replayByCase(goldens, corpus);
  return async function embed(text, opts) {
    const vec = Float32Array.from(await inner(text, opts));
    vec[dim] += delta;
    return vec;
  };
}

/**
 * 2. Drop the prefix — return the `none`-prefix golden of the same text
 *    when a prefixed case is requested. Same weights, wrong task token.
 */
export function dropPrefixEmbedder(goldens, corpus) {
  const map = indexByCase(goldens, corpus);
  const noneByText = new Map();
  for (const c of corpus.cases) {
    if (c.prefix === "none") {
      const hit = map.get(caseKey(c.text, "none"));
      if (hit) noneByText.set(c.text, hit.item);
    }
  }
  return async function embed(text, { prefix } = {}) {
    if (prefix === "document" || prefix === "query") {
      const dropped = noneByText.get(text);
      if (dropped) return asFloat32(dropped.vector);
    }
    const hit = map.get(caseKey(text, prefix));
    if (!hit) throw new Error("dropPrefixEmbedder: no golden");
    return asFloat32(hit.item.vector);
  };
}

/**
 * 3. Swap pooling — return the committed CLS-pooled control vector for the
 *    targeted case (generated with llama.cpp `--pooling cls` at pin time).
 *    Other cases replay correctly so the receipt names the swapped item.
 */
export function swapPoolingEmbedder(goldens, corpus, control) {
  if (!control?.id || !Array.isArray(control.vector)) {
    throw new Error("swapPoolingEmbedder: missing committed CLS-pooled control vector");
  }
  const map = indexByCase(goldens, corpus);
  const byId = goldensById(goldens);
  const target = byId.get(control.id);
  return async function embed(text, { prefix } = {}) {
    const hit = map.get(caseKey(text, prefix));
    if (hit && hit.case.id === control.id) return asFloat32(control.vector);
    if (target && text === undefined) return asFloat32(control.vector);
    if (!hit) throw new Error("swapPoolingEmbedder: no golden");
    return asFloat32(hit.item.vector);
  };
}

/** 4. Truncate a dimension — 768 → 767. */
export function truncateEmbedder(goldens, corpus) {
  const inner = replayByCase(goldens, corpus);
  return async function embed(text, opts) {
    const vec = await inner(text, opts);
    return vec.subarray(0, vec.length - 1);
  };
}

export function applyWrong(kind, goldens, extras = {}) {
  const corpus = extras.corpus;
  switch (kind) {
    case "perturb":
      return perturbEmbedder(goldens, corpus);
    case "drop-prefix":
      return dropPrefixEmbedder(goldens, corpus);
    case "swap-pooling":
      return swapPoolingEmbedder(goldens, corpus, extras.control);
    case "truncate":
      return truncateEmbedder(goldens, corpus);
    default:
      throw new Error(`unknown wrong-embedder kind: ${kind}`);
  }
}
