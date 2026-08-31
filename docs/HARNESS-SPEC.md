# Milton conformance + bench harness — spec

**The keystone artifact.** Built before the embedder. It is three things at once: the
correctness **oracle** (is a vector right?), the optimization **target** (is it lighter/
faster?), and the agent's **iteration signal** (am I converging?). Milton is not "done"
until it clears this gate; the gate is not "passing" until it has been shown to FAIL on a
deliberately-wrong embedding.

## The reference (the oracle)

- **v1 oracle:** nomic-embed-text-v1.5, GGUF, via **llama.cpp** — the exact path Flair
  uses today (Harper `models.embed` → harper-fabric-embeddings → llama.cpp). Mean
  pooling and the rest of the arch come from the file, not from a nomic-shaped
  hardcode. The engine is GGUF-driven; this pin is verified-first. Do not loosen
  epsilon.
- **Prefix convention:** config, not code. v1's config is nomic's `search_query:` /
  `search_document:` prefixes, applied as Flair applies them. The harness must
  replicate the configured prefixes exactly — a prefix mismatch is a silent recall
  bug, not a rounding difference. A second BERT-family file (bge / gte / e5 /
  nomic-v2) is a new golden pin + prefix config, not a harness rewrite.
- **Reference vectors are generated once, pinned, and committed** (or content-addressed):
  for a fixed conformance corpus, run the reference and store `{text, prefix, vector}`.
  These are the golden set. Pin the exact GGUF file digest + llama.cpp commit that
  produced them — an undated/unpinned reference is not a reference.

## The conformance corpus

- A fixed, committed set covering: short/long text, unicode, whitespace/empty edges,
  the prefix variants, and a few real Flair-shaped memory strings.
- Small enough to run every CI push, broad enough that a wrong tokenizer/pooling/prefix
  path can't slip through. Document what each case is *for* (each is a trap for a specific
  failure mode).

## The gate

- For each corpus item: `cosine(milton_vector, reference_vector) >= 1 - EPSILON` AND
  per-dimension `max_abs_diff <= EPSILON_ABS`. Both, not either — cosine alone hides a
  scale bug; abs-diff alone is noisy on near-orthogonal.
- `EPSILON` / `EPSILON_ABS` are set from observed float determinism of the reference
  itself (run the reference twice, measure its own run-to-run delta, set the gate a small
  margin above that floor). **Record the number and how it was derived** — a tolerance
  pulled from the air is a decorative gate.
- **Receipts:** every run emits `{corpus_digest, reference_digest, n, max_cos_dist,
  mean_cos_dist, max_abs, pass|fail, per-failure {text, expected, got, delta}}`. Camelid-
  style: the evidence is the output, not a green checkmark.
- **Fail closed:** any item over tolerance → the whole run FAILS and names which items.

## The must-fail control (before the gate is trusted)

The gate MUST be shown to reject a deliberately-wrong embedder before it is allowed to
grade a real one — perturb a vector, drop the prefix, swap pooling, truncate a dimension;
each must turn the run RED and name the failure. A conformance suite that has only ever
passed is a decoration. This control ships as a test.

## The bench (the optimization target)

Alongside correctness, measure on the same corpus/host:
- **Install footprint**: installed package size (MB), and whether any native binary is
  pulled into `src/` (must be zero — reference toolchain lives in `harness/` only).
- **Cold-start**: time to first embedding (model load + init).
- **Throughput**: embeddings/sec, single and batched.
- **Baseline**: the current Flair path (Harper models.embed) measured on the same host,
  so "lighter/faster" is a delta against a number we measured, not a claim.

## Done-condition for the harness (this issue)

- Reference vectors generated + pinned (digests recorded).
- Gate implemented (cosine + abs, derived epsilon, receipts, fail-closed).
- Must-fail control passing (i.e. it makes a wrong embedder RED).
- Baseline bench numbers captured for the current Flair path.
- No embedder yet — that's the next issue, and it iterates against this.
