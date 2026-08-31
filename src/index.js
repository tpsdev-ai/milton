/**
 * Public API surface for @tpsdev-ai/milton.
 *
 * Tokenizer slice lives in `crate/` (Rust, Refs #3). The embedder (forward
 * pass) is not implemented yet and is not "done" until it clears the
 * golden-vector gate in harness/. An unverified path must refuse, never guess.
 * Do not add a JS tokenize here — that would be a second, unverified path.
 */
export function embed() {
  throw new Error(
    "milton embedder is not implemented yet — tokenizer is crate/ (Refs #3); forward pass is a later issue",
  );
}
