import { compareVectors } from "./metrics.js";
import { applyPrefix } from "./prefix.js";
import { asFloat32, goldensById, loadEpsilon, loadGoldens, loadPin, referenceDigest } from "./goldens.js";
import { corpusDigest, loadCorpus } from "./corpus.js";
import { buildReceipt } from "./receipts.js";

/**
 * Run the golden-vector gate.
 *
 * `embed` is `(text, { prefix }) => Promise<ArrayLike<number>>`.
 * Fail closed: any item over tolerance (or any throw / missing golden /
 * dim mismatch) fails the whole run and is named in the receipt.
 */
export async function runGate(embed, options = {}) {
  const corpus = options.corpus ?? loadCorpus();
  const goldens = options.goldens ?? loadGoldens();
  const pin = options.pin ?? loadPin();
  const eps = options.epsilonFile ?? loadEpsilon();
  const epsilon = options.epsilon ?? eps.epsilon;
  const epsilonAbs = options.epsilonAbs ?? eps.epsilon_abs;
  const byId = goldensById(goldens);

  if (corpusDigest(corpus) !== goldens.corpus_digest) {
    throw new Error(
      `fail-closed: corpus digest ${corpusDigest(corpus)} != goldens.corpus_digest ${goldens.corpus_digest}`,
    );
  }

  const comparisons = [];
  for (const c of corpus.cases) {
    const golden = byId.get(c.id);
    if (!golden || !Array.isArray(golden.vector)) {
      comparisons.push({
        id: c.id,
        text: c.text,
        prefix: c.prefix,
        pass: false,
        reason: "missing_golden",
        expected: null,
        got: null,
        cos_dist: null,
        max_abs: null,
      });
      continue;
    }
    const expected = asFloat32(golden.vector);
    let got;
    try {
      got = await embed(c.text, { prefix: c.prefix, prefixed: applyPrefix(c.text, c.prefix) });
    } catch (err) {
      comparisons.push({
        id: c.id,
        text: c.text,
        prefix: c.prefix,
        pass: false,
        reason: `embed_threw:${err?.message ?? String(err)}`,
        expected,
        got: null,
        cos_dist: null,
        max_abs: null,
      });
      continue;
    }
    if (!got || typeof got.length !== "number") {
      comparisons.push({
        id: c.id,
        text: c.text,
        prefix: c.prefix,
        pass: false,
        reason: "embed_returned_empty",
        expected,
        got,
        cos_dist: null,
        max_abs: null,
      });
      continue;
    }
    const cmp = compareVectors(got, expected, { epsilon, epsilonAbs });
    comparisons.push({
      id: c.id,
      text: c.text,
      prefix: c.prefix,
      expected,
      got,
      ...cmp,
    });
  }

  return buildReceipt({
    corpus_digest: corpusDigest(corpus),
    reference_digest: referenceDigest(goldens),
    pin,
    epsilon,
    epsilon_abs: epsilonAbs,
    comparisons,
  });
}
