/**
 * Camelid-style receipts: the evidence is the output, not a green checkmark.
 *
 * Shape (HARNESS-SPEC.md):
 *   {corpus_digest, reference_digest, n, max_cos_dist, mean_cos_dist,
 *    max_abs, pass|fail, per-failure {text, expected, got, delta}}
 */

const PREVIEW = 8;

function preview(vec) {
  if (!vec || typeof vec.length !== "number") return { dims: 0, head: [] };
  return {
    dims: vec.length,
    head: Array.from(vec).slice(0, PREVIEW),
  };
}

export function buildReceipt({
  corpus_digest,
  reference_digest,
  pin,
  epsilon,
  epsilon_abs,
  comparisons,
}) {
  const n = comparisons.length;
  let maxCosDist = 0;
  let sumCosDist = 0;
  let maxAbs = 0;
  const failures = [];

  for (const c of comparisons) {
    const cosDist = c.cos_dist ?? 1;
    const abs = c.max_abs ?? Number.POSITIVE_INFINITY;
    if (cosDist > maxCosDist) maxCosDist = cosDist;
    if (abs > maxAbs && Number.isFinite(abs)) maxAbs = abs;
    sumCosDist += Number.isFinite(cosDist) ? cosDist : 1;
    if (!c.pass) {
      failures.push({
        id: c.id,
        text: c.text,
        prefix: c.prefix,
        expected: preview(c.expected),
        got: preview(c.got),
        delta: {
          cos_dist: c.cos_dist,
          max_abs: c.max_abs,
          reason: c.reason,
        },
      });
    }
  }

  const pass = failures.length === 0;
  return {
    schema: "milton.receipt/1",
    result: pass ? "pass" : "fail",
    n,
    failed: failures.length,
    max_cos_dist: maxCosDist,
    mean_cos_dist: n ? sumCosDist / n : 0,
    max_abs: maxAbs,
    epsilon,
    epsilon_abs,
    corpus_digest,
    reference_digest,
    pin: pin
      ? {
          gguf_sha256: pin.gguf_sha256,
          llamacpp_commit: pin.llamacpp_commit,
          llamacpp_digest: pin.llamacpp_digest,
          pooling: pin.pooling,
          embd_normalize: pin.embd_normalize,
        }
      : undefined,
    failures,
  };
}

export function formatReceipt(receipt) {
  const lines = [
    `milton-gate ${receipt.result.toUpperCase()}  n=${receipt.n}  failed=${receipt.failed}`,
    `  corpus_digest     ${receipt.corpus_digest}`,
    `  reference_digest  ${receipt.reference_digest}`,
    `  max_cos_dist      ${receipt.max_cos_dist}`,
    `  mean_cos_dist     ${receipt.mean_cos_dist}`,
    `  max_abs           ${receipt.max_abs}`,
    `  epsilon           ${receipt.epsilon}   epsilon_abs ${receipt.epsilon_abs}`,
  ];
  if (receipt.pin) {
    lines.push(`  gguf_sha256       ${receipt.pin.gguf_sha256}`);
    lines.push(`  llamacpp_commit   ${receipt.pin.llamacpp_commit}`);
  }
  for (const f of receipt.failures) {
    lines.push(
      `  FAIL ${f.id}  prefix=${f.prefix}  ${f.delta.reason}  text=${JSON.stringify(f.text).slice(0, 80)}`,
    );
  }
  return lines.join("\n");
}
