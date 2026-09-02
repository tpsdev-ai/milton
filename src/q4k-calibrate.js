/**
 * Load-time Q4_K {per-k | all-k} calibration.
 *
 * Both variants are bit-exact (same integer tree, same f32 stage). The
 * choice cannot change results — only time. No engine / CPU sniffing.
 *
 * Work bound: one superblock × 32 tokens, both variants, plus n=1 so the
 * crossover is measured rather than assumed. Threshold rule is documented
 * in the #46 plan-first comment.
 */

const WARMUP = 4;
const SAMPLES = 16;

function median(xs) {
  const a = xs.slice().sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

function timeCalls(fn, n) {
  const samples = [];
  for (let i = 0; i < WARMUP; i += 1) fn();
  for (let i = 0; i < SAMPLES; i += 1) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  return { n, median_ms: median(samples), samples };
}

export function crossoverThreshold(perk1, perk32, allk1, allk32) {
  // Fail closed on time: if all-k does not win the full tile, never pick it.
  if (!(allk32 < perk32)) return 33;
  if (!(allk1 > perk1)) return 1;
  const dp = perk32 - perk1;
  const da = allk32 - allk1;
  const denom = da - dp;
  if (!(denom < 0) || !Number.isFinite(denom)) return 33;
  const n = 1 + (31 * (perk1 - allk1)) / denom;
  if (!Number.isFinite(n)) return 33;
  return Math.min(32, Math.max(1, Math.ceil(n)));
}

/**
 * @param {{
 *   q4kSetForce: (name: string) => void,
 *   q4kSetThreshold: (t: number) => void,
 *   q4kRunPerk: (n: number) => void,
 *   q4kRunAllk: (n: number) => void,
 *   q4kVariantMaxAbs: () => number,
 *   q4kThreshold: () => number,
 * }} api
 * @param {NodeJS.ProcessEnv} [env]
 */
export function applyQ4kPolicy(api, env = process.env) {
  const raw = (env.MILTON_Q4K_VARIANT || "").toLowerCase();
  if (raw === "perk" || raw === "per-k" || raw === "per_k") {
    api.q4kSetForce("perk");
    return {
      mode: "perk",
      forced: true,
      threshold: api.q4kThreshold(),
      cost_ms: 0,
      max_abs: null,
    };
  }
  if (raw === "allk" || raw === "all-k" || raw === "all_k") {
    api.q4kSetForce("allk");
    return {
      mode: "allk",
      forced: true,
      threshold: api.q4kThreshold(),
      cost_ms: 0,
      max_abs: null,
    };
  }

  const t0 = performance.now();
  const maxAbs = api.q4kVariantMaxAbs();
  if (!(maxAbs === 0)) {
    throw new Error(
      `fail-closed: Q4_K per-k vs all-k not bit-exact max_abs=${maxAbs}`,
    );
  }
  const perk1 = timeCalls(() => api.q4kRunPerk(1), 1);
  const perk32 = timeCalls(() => api.q4kRunPerk(32), 32);
  const allk1 = timeCalls(() => api.q4kRunAllk(1), 1);
  const allk32 = timeCalls(() => api.q4kRunAllk(32), 32);
  const threshold = crossoverThreshold(
    perk1.median_ms,
    perk32.median_ms,
    allk1.median_ms,
    allk32.median_ms,
  );
  api.q4kSetForce("auto");
  api.q4kSetThreshold(threshold);
  const costMs = performance.now() - t0;
  return {
    mode: "auto",
    forced: false,
    threshold,
    cost_ms: costMs,
    max_abs: maxAbs,
    perk_n1_ms: perk1.median_ms,
    perk_n32_ms: perk32.median_ms,
    allk_n1_ms: allk1.median_ms,
    allk_n32_ms: allk32.median_ms,
  };
}
