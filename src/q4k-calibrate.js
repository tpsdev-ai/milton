/**
 * Load-time Q4_K calibration framework.
 *
 * WARMUP=100 (Kern #47): V8 Liftoff → TurboFan is invocation-driven.
 * The auto path **times** the shipped variant(s) — it does not short-circuit
 * to threshold 33 without measuring. All-k (`Q4kUnpacked`) is not in the
 * wasm, so the only timed kernel is per-k; threshold stays 33 after the
 * measurement. Crossover math is unchanged for a later letter ((b′)).
 *
 * The report carries median(first 5 warmup calls) vs median(last 5) so
 * a Liftoff → TurboFan step is visible. Calibration cannot change
 * results — only time. No engine / CPU sniffing.
 */

const WARMUP = 100;
const SAMPLES = 8;

function median(xs) {
  const a = xs.slice().sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

/**
 * Time `fn` with WARMUP timed calls (not discarded) then SAMPLES.
 * first5 / last5 are medians of the warmup window — equal means no
 * observable tier-up during those 100 invocations.
 */
function timeCalls(fn, n) {
  const warmup = [];
  for (let i = 0; i < WARMUP; i += 1) {
    const t0 = performance.now();
    fn();
    warmup.push(performance.now() - t0);
  }
  const samples = [];
  for (let i = 0; i < SAMPLES; i += 1) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  const first5_ms = median(warmup.slice(0, 5));
  const last5_ms = median(warmup.slice(-5));
  const equal = first5_ms === last5_ms;
  let verdict;
  if (equal) verdict = "equal / no tier-up";
  else if (last5_ms < first5_ms) verdict = "tier-up";
  else verdict = "no tier-up";
  return {
    n,
    median_ms: median(samples),
    samples,
    first5_ms,
    last5_ms,
    equal,
    verdict,
  };
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
 *   q4kRunPerk: (n: number) => void,
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

  if (typeof api.q4kRunPerk !== "function") {
    throw new Error("fail-closed: Q4_K auto path requires q4kRunPerk — will not guess a threshold");
  }

  const t0 = performance.now();
  // Time the shipped variant. Do not write threshold 33 until after the bench.
  const perk32 = timeCalls(() => api.q4kRunPerk(32), 32);
  const perk1 = timeCalls(() => api.q4kRunPerk(1), 1);
  // Only perk ships — no second variant to cross. Threshold 33 after measuring.
  const threshold = 33;
  api.q4kSetForce("auto");
  api.q4kSetThreshold(threshold);
  return {
    mode: "auto",
    forced: false,
    threshold: api.q4kThreshold(),
    cost_ms: performance.now() - t0,
    max_abs: null,
    shipped_variants: ["perk"],
    perk_n1_ms: perk1.median_ms,
    perk_n32_ms: perk32.median_ms,
    allk_n1_ms: null,
    allk_n32_ms: null,
    perk_n32_first5_ms: perk32.first5_ms,
    perk_n32_last5_ms: perk32.last5_ms,
    perk_n32_warmup_equal: perk32.equal,
    perk_n32_warmup_verdict: perk32.verdict,
    perk_n1_first5_ms: perk1.first5_ms,
    perk_n1_last5_ms: perk1.last5_ms,
    perk_n1_warmup_equal: perk1.equal,
    perk_n1_warmup_verdict: perk1.verdict,
  };
}

export { WARMUP, SAMPLES, timeCalls };
