/**
 * Gate metrics. Both required:
 *   cosine(got, expected) >= 1 - EPSILON
 *   max_i |got_i - expected_i| <= EPSILON_ABS
 *
 * Cosine alone hides a scale bug; abs-diff alone is noisy on near-orthogonal.
 */

export function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error(`cosine: length mismatch ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return na === 0 && nb === 0 ? 1 : 0;
  return dot / denom;
}

export function maxAbsDiff(a, b) {
  if (a.length !== b.length) {
    throw new Error(`maxAbsDiff: length mismatch ${a.length} vs ${b.length}`);
  }
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

export function l2Normalize(vec) {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return Float32Array.from(vec);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

/**
 * @param {ArrayLike<number>} got
 * @param {ArrayLike<number>} expected
 * @param {{ epsilon: number, epsilonAbs: number }} tol
 */
export function compareVectors(got, expected, tol) {
  if (got.length !== expected.length) {
    return {
      pass: false,
      reason: `dim_mismatch:${expected.length}->${got.length}`,
      cosine: null,
      cos_dist: null,
      max_abs: null,
    };
  }
  const cos = cosine(got, expected);
  const cosDist = Math.max(0, 1 - cos);
  const maxAbs = maxAbsDiff(got, expected);
  const pass = cos >= 1 - tol.epsilon && maxAbs <= tol.epsilonAbs;
  return {
    pass,
    reason: pass
      ? null
      : [
          cos < 1 - tol.epsilon ? `cos_dist=${cosDist}` : null,
          maxAbs > tol.epsilonAbs ? `max_abs=${maxAbs}` : null,
        ]
          .filter(Boolean)
          .join(","),
    cosine: cos,
    cos_dist: cosDist,
    max_abs: maxAbs,
  };
}
