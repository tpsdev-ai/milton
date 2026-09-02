/* tslint:disable */
/* eslint-disable */
/**
 * Harness / JS glue: force a Q4_K inner-loop variant (`perk` | `allk` | `auto`).
 * Does not live in the wasm as an env-var string — JS reads `MILTON_Q4K_VARIANT`.
 */
export function q4kSetForce(name: string): void;
export function q4kSetThreshold(t: number): void;
export function q4kThreshold(): number;
/**
 * One synthetic superblock × `n_tokens` of the per-k variant (calibrator).
 */
export function q4kRunPerk(n_tokens: number): void;
/**
 * One synthetic superblock × `n_tokens` of the all-k variant (calibrator).
 */
export function q4kRunAllk(n_tokens: number): void;
/**
 * Bit-exact check: `max_abs` between the two variants on the synth tile.
 */
export function q4kVariantMaxAbs(): number;
/**
 * In-process embedder loaded from GGUF bytes.
 */
export class Milton {
  free(): void;
  /**
   * `gguf` is the raw nomic-embed-text-v1.5 GGUF (architecture from the file).
   */
  constructor(gguf: Uint8Array);
  /**
   * `embed(text, prefix) -> Float32Array`. Prefix kind is `document` |
   * `query` | `none`. Templates (`search_document: ` / `search_query: `)
   * are config on the Rust side — space after the colon is load-bearing.
   */
  embed(text: string, prefix: string): Float32Array;
  embedWithFault(text: string, prefix: string, fault: string): Float32Array;
  embeddingLength(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_milton_free: (a: number, b: number) => void;
  readonly milton_new: (a: number, b: number) => [number, number, number];
  readonly milton_embed: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly milton_embedWithFault: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
  readonly milton_embeddingLength: (a: number) => number;
  readonly q4kSetForce: (a: number, b: number) => void;
  readonly q4kSetThreshold: (a: number) => void;
  readonly q4kThreshold: () => number;
  readonly q4kRunPerk: (a: number) => void;
  readonly q4kRunAllk: (a: number) => void;
  readonly q4kVariantMaxAbs: () => number;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
