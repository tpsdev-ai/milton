/**
 * Public API surface for @tpsdev-ai/milton.
 *
 * Intentionally empty in this PR (issue #1 is harness-only). The embedder
 * lands in a later issue and is not "done" until it clears the golden-vector
 * gate in harness/. An unverified path must refuse, never guess.
 */
export function embed() {
  throw new Error(
    "milton embedder is not implemented yet — this package ships the conformance harness first (see harness/, Refs #1)",
  );
}
