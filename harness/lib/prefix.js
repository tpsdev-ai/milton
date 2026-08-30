/**
 * Flair's nomic prefix convention — replicated exactly.
 *
 * Source of truth (verified 2026-08-30 against public trees):
 * - Flair `resources/embeddings-provider.ts`: gate ON; callers pass
 *   `inputType: 'document'` for stored Memory.content, `'query'` for
 *   SemanticSearch `q`. Omitted inputType is passthrough (no prefix).
 * - harper-fabric-embeddings `src/engine.ts` `NOMIC_TEMPLATES`:
 *     document: 'search_document: {text}'
 *     query:    'search_query: {text}'
 *   Space after the colon is load-bearing. Passing the prefix STRING
 *   (`'search_document'`) as the inputType VALUE is a known silent inversion
 *   bug — this helper only accepts the closed union below.
 *
 * A prefix mismatch is a silent recall bug, not a rounding difference.
 */

export const PREFIX = Object.freeze({
  document: "search_document: ",
  query: "search_query: ",
  none: "",
});

export const PREFIX_KINDS = Object.freeze(["document", "query", "none"]);

/**
 * @param {string} text
 * @param {'document' | 'query' | 'none'} kind
 * @returns {string}
 */
export function applyPrefix(text, kind) {
  if (kind !== "document" && kind !== "query" && kind !== "none") {
    throw new Error(
      `applyPrefix: invalid kind ${JSON.stringify(kind)} (expected 'document' | 'query' | 'none')`,
    );
  }
  if (typeof text !== "string") {
    throw new Error(`applyPrefix: text must be a string, got ${typeof text}`);
  }
  return PREFIX[kind] + text;
}
