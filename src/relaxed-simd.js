/**
 * Relaxed-SIMD capability probe + kernel pick (issue #43).
 *
 * Detection is `WebAssembly.validate` of a one-function module that
 * contains `i16x8.relaxed_dot_i8x16_i7x16_s` (0xfd 0x112). Never a
 * Node / bun version sniff — bun 1.3.10 validates SIMD128 and rejects
 * this probe (Nathan 2026-09-03).
 *
 * Pick is once at load. `probe` is the capability, not the pick:
 * `MILTON_RELAXED_SIMD=0` on Node ≥22 is
 * `{kernel:'simd128', probe:true, forced:true}`.
 */

/** (module (func (param v128 v128) (result v128)
 *    local.get 0 local.get 1 i16x8.relaxed_dot_i8x16_i7x16_s)) */
export const RELAXED_DOT_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
  0x02, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0b, 0x01,
  0x09, 0x00, 0x20, 0x00, 0x20, 0x01, 0xfd, 0x92, 0x02, 0x0b,
]);

/** SIMD128 control: same shape with `i8x16.add` (0xfd 0x6e). */
export const SIMD128_ADD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
  0x02, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01,
  0x08, 0x00, 0x20, 0x00, 0x20, 0x01, 0xfd, 0x6e, 0x0b,
]);

/**
 * Capability only — not a loader decision.
 * @param {typeof globalThis} [global]
 */
export function probeRelaxedSimd(global = globalThis) {
  if (typeof global.WebAssembly?.validate !== "function") return false;
  try {
    return global.WebAssembly.validate(RELAXED_DOT_PROBE);
  } catch {
    return false;
  }
}

/**
 * @param {NodeJS.ProcessEnv} [env]
 * @param {typeof globalThis} [global]
 * @returns {{ kernel: 'relaxed' | 'simd128', probe: boolean, forced: boolean }}
 */
export function resolveQmatmulKernel(env = process.env, global = globalThis) {
  const probe = probeRelaxedSimd(global);
  const raw = env.MILTON_RELAXED_SIMD;
  if (raw === "0") {
    return { kernel: "simd128", probe, forced: true };
  }
  if (raw === "1") {
    if (!probe) {
      throw new Error(
        "fail-closed: MILTON_RELAXED_SIMD=1 but WebAssembly.validate rejected the relaxed-dot probe",
      );
    }
    return { kernel: "relaxed", probe, forced: true };
  }
  return { kernel: probe ? "relaxed" : "simd128", probe, forced: false };
}
