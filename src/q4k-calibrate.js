/**
 * Load-time Q4_K calibration framework.
 *
 * WARMUP=100 (Kern #47): V8 Liftoff → TurboFan is invocation-driven.
 * Kept so a later second variant ((b′) lane-wise scale) measures the
 * optimized tier. All-k (`Q4kUnpacked`) is not shipped — after the
 * tier fix, `--no-liftoff` on this VM and M4 both stay at threshold 33
 * (always per-k). Auto path therefore records threshold 33 and does
 * not time a missing kernel. Crossover math is unchanged for (b′).
 *
 * Calibration cannot change results — only time. No engine / CPU sniffing.
 */

const WARMUP = 100;
const SAMPLES = 8;

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
  // Fail closed on time: if the second variant does not win the full tile, never pick it.
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
 *   q4kRunPerk?: (n: number) => void,
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
      shipped_variants: ["perk"],
    };
  }
  if (raw === "allk" || raw === "all-k" || raw === "all_k") {
    throw new Error(
      "fail-closed: Q4_K all-k is not shipped — optimized-tier pick is always per-k",
    );
  }

  // No second variant in the wasm. Framework stays; do not time a missing kernel.
  const t0 = performance.now();
  api.q4kSetForce("auto");
  api.q4kSetThreshold(33);
  return {
    mode: "auto",
    forced: false,
    threshold: api.q4kThreshold(),
    cost_ms: performance.now() - t0,
    max_abs: null,
    shipped_variants: ["perk"],
    perk_n1_ms: null,
    perk_n32_ms: null,
    allk_n1_ms: null,
    allk_n32_ms: null,
  };
}

export { WARMUP, SAMPLES, timeCalls };
