/**
 * Host-tuned A2 serial→parallel token gate (issue #59).
 *
 * Default is the #57 KVM-host literal 32. `MILTON_ATTN_MIN_TOKENS` overrides
 * the effective value. Non-numeric / out-of-range falls back to 32 (never
 * throws); clamp warns once, same grain as `MILTON_THREADS` (#51).
 * JS reads the env; wasm stores the applied value (`attnSetMinTokens`).
 * This is (a) only — not a measured load-time crossover.
 */

export const ATTN_MIN_TOKENS_DEFAULT = 32;
export const ATTN_MIN_TOKENS_MAX = 8192;

let clampWarned = false;

function warnAttnClamp(raw, applied) {
  if (clampWarned) return;
  clampWarned = true;
  console.warn(`MILTON_ATTN_MIN_TOKENS=${raw} is out of range; using ${applied}`);
}

/** `MILTON_ATTN_MIN_TOKENS` or 32. Out-of-range / non-numeric → 32. */
export function resolveAttnMinTokens(env = process.env) {
  const raw = env.MILTON_ATTN_MIN_TOKENS;
  if (raw === undefined || raw === "") {
    return ATTN_MIN_TOKENS_DEFAULT;
  }
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    warnAttnClamp(raw, ATTN_MIN_TOKENS_DEFAULT);
    return ATTN_MIN_TOKENS_DEFAULT;
  }
  const want = Math.floor(n);
  if (want < 1 || want > ATTN_MIN_TOKENS_MAX) {
    warnAttnClamp(raw, ATTN_MIN_TOKENS_DEFAULT);
    return ATTN_MIN_TOKENS_DEFAULT;
  }
  return want;
}

/**
 * Push the resolved gate into wasm and return the value the crate stored.
 * Fail-closed if the tiny export is missing (stale blob).
 */
export function applyAttnMinTokens(api, env = process.env) {
  const gate = resolveAttnMinTokens(env);
  if (typeof api?.attnSetMinTokens !== "function" || typeof api?.attnMinTokens !== "function") {
    throw new Error(
      "fail-closed: wasm missing attnMinTokens export — rebuild with remapped scripts/build-wasm.sh",
    );
  }
  api.attnSetMinTokens(gate);
  return api.attnMinTokens() >>> 0;
}
