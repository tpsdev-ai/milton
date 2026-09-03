/**
 * Load-time Q4_K calibration framework.
 *
 * WARMUP=100 (Kern #47): V8 Liftoff → TurboFan is invocation-driven.
 * Auto times both shipped variants — perk and (b′) — at n=1 and n=32,
 * then `crossoverThreshold`. Fail-closed: if (b′) does not win the full
 * tile (n=32), threshold stays 33 (never pick it). All-k is not in the
 * wasm. Crossover math is unchanged; the second-variant pair is now
 * bprime timings.
 *
 * The report carries median(first 5 warmup calls) vs median(last 5) so
 * a Liftoff → TurboFan step is visible. Calibration cannot change
 * results — only time. No engine / CPU sniffing.
 */

const WARMUP = 100;
const SAMPLES = 8;

const BPRIME_ALIASES = new Set(["bprime", "b-prime", "b_prime", "b'"]);
const PERK_ALIASES = new Set(["perk", "per-k", "per_k"]);
const ALLK_ALIASES = new Set(["allk", "all-k", "all_k"]);

function median(xs) {
  const a = xs.slice().sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

function nowFn(api) {
  return typeof api.now === "function" ? api.now : () => performance.now();
}

/**
 * Time `fn` with WARMUP timed calls (not discarded) then SAMPLES.
 * first5 / last5 are medians of the warmup window — equal means no
 * observable tier-up during those 100 invocations.
 */
function timeCalls(fn, n, now = () => performance.now()) {
  const warmup = [];
  for (let i = 0; i < WARMUP; i += 1) {
    const t0 = now();
    fn();
    warmup.push(now() - t0);
  }
  const samples = [];
  for (let i = 0; i < SAMPLES; i += 1) {
    const t0 = now();
    fn();
    samples.push(now() - t0);
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

function warmupFields(prefix, timed) {
  return {
    [`${prefix}_n${timed.n}_ms`]: timed.median_ms,
    [`${prefix}_n${timed.n}_first5_ms`]: timed.first5_ms,
    [`${prefix}_n${timed.n}_last5_ms`]: timed.last5_ms,
    [`${prefix}_n${timed.n}_warmup_equal`]: timed.equal,
    [`${prefix}_n${timed.n}_warmup_verdict`]: timed.verdict,
  };
}

/**
 * @param {{
 *   q4kSetForce: (name: string) => void,
 *   q4kSetThreshold: (t: number) => void,
 *   q4kRunPerk: (n: number) => void,
 *   q4kRunBprime: (n: number) => void,
 *   q4kThreshold: () => number,
 *   now?: () => number,
 * }} api
 * @param {NodeJS.ProcessEnv} [env]
 */
export function applyQ4kPolicy(api, env = process.env) {
  const raw = (env.MILTON_Q4K_VARIANT || "").toLowerCase();
  if (PERK_ALIASES.has(raw)) {
    api.q4kSetForce("perk");
    return {
      mode: "perk",
      forced: true,
      threshold: api.q4kThreshold(),
      cost_ms: 0,
      max_abs: null,
      shipped_variants: ["perk", "bprime"],
    };
  }
  if (BPRIME_ALIASES.has(raw)) {
    api.q4kSetForce("bprime");
    return {
      mode: "bprime",
      forced: true,
      threshold: api.q4kThreshold(),
      cost_ms: 0,
      max_abs: null,
      shipped_variants: ["perk", "bprime"],
    };
  }
  if (ALLK_ALIASES.has(raw)) {
    console.warn(
      `MILTON_Q4K_VARIANT=${raw} is not shipped; falling through to auto/per-k`,
    );
  }

  if (typeof api.q4kRunPerk !== "function") {
    throw new Error("fail-closed: Q4_K auto path requires q4kRunPerk — will not guess a threshold");
  }
  if (typeof api.q4kRunBprime !== "function") {
    throw new Error(
      "fail-closed: Q4_K auto path requires q4kRunBprime — will not guess a threshold",
    );
  }

  const now = nowFn(api);
  const wall0 = performance.now();
  const perk32 = timeCalls(() => api.q4kRunPerk(32), 32, now);
  const perk1 = timeCalls(() => api.q4kRunPerk(1), 1, now);
  const bprime32 = timeCalls(() => api.q4kRunBprime(32), 32, now);
  const bprime1 = timeCalls(() => api.q4kRunBprime(1), 1, now);
  const threshold = crossoverThreshold(
    perk1.median_ms,
    perk32.median_ms,
    bprime1.median_ms,
    bprime32.median_ms,
  );
  api.q4kSetForce("auto");
  api.q4kSetThreshold(threshold);
  return {
    mode: "auto",
    forced: false,
    threshold: api.q4kThreshold(),
    cost_ms: performance.now() - wall0,
    max_abs: null,
    shipped_variants: ["perk", "bprime"],
    perk_n1_ms: perk1.median_ms,
    perk_n32_ms: perk32.median_ms,
    allk_n1_ms: null,
    allk_n32_ms: null,
    bprime_n1_ms: bprime1.median_ms,
    bprime_n32_ms: bprime32.median_ms,
    ...warmupFields("perk", perk32),
    ...warmupFields("perk", perk1),
    ...warmupFields("bprime", bprime32),
    ...warmupFields("bprime", bprime1),
  };
}

export { WARMUP, SAMPLES, timeCalls };
