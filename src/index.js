/**
 * Public API: `embed(text, {prefix}) -> Float32Array`.
 *
 * WASM-SIMD path (Refs #6). Same Rust `Model` as the native bins, compiled
 * with +simd128. Prefix is config (`document` | `query` | `none`); templates
 * are `search_document: ` / `search_query: ` / passthrough (load-bearing space).
 *
 * Two artifacts, one loader (Refs #44). The shared-memory module is used
 * only when SharedArrayBuffer + Atomics exist and WebAssembly.validate
 * accepts a shared-memory probe. Absence of SAB is the ordinary path —
 * `wasm/milton_bg.wasm` — not an error.
 *
 * Fail-closed: missing prebuilt wasm, missing GGUF, or an unverified path
 * refuses. No native compile and no per-platform build at install.
 * The reference toolchain stays in harness/ as the oracle.
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { applyQ4kPolicy } from "./q4k-calibrate.js";
import {
  canUseWasmThreads,
  resolveThreadCount,
  startWorkerPool,
} from "./wasm-threads.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const WASM_PATH = join(ROOT, "wasm", "milton_bg.wasm");
const WASM_THREADS_PATH = join(ROOT, "wasm", "milton_threads_bg.wasm");

const PREFIX_KINDS = new Set(["document", "query", "none"]);

export function resolveWasmPath() {
  return WASM_PATH;
}

export function resolveThreadsWasmPath() {
  return WASM_THREADS_PATH;
}

export function resolveGguf(env = process.env) {
  return (
    env.MILTON_GGUF ||
    env.MILTON_REFERENCE_GGUF ||
    join(ROOT, "harness", "vendor", "models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
  );
}

function resolveKind(prefix) {
  if (typeof prefix === "string") return prefix;
  if (prefix && typeof prefix === "object" && typeof prefix.prefix === "string") {
    return prefix.prefix;
  }
  throw new Error(
    "fail-closed: embed(text, prefix) requires prefix 'document' | 'query' | 'none'",
  );
}

function resolveFault(prefix) {
  if (prefix && typeof prefix === "object" && typeof prefix.fault === "string") {
    return prefix.fault;
  }
  return null;
}

let wasmReady = null;
let instance = null;
let queue = Promise.resolve();
let api = null;
/** Retain Worker handles so GC cannot terminate parked threads. */
let workerPool = null;

/** Last load-time Q4_K calibration (or forced-variant) report. */
export let lastQ4kCalibration = null;

/** `'single' | 'threads'` — which prebuilt artifact the loader picked. */
export let lastWasmArtifact = null;

/** Pool size after load. `1` on the single-thread artifact. */
export let lastThreadCount = 1;

export { canUseWasmThreads, resolveThreadCount };

function ensureWasm() {
  if (wasmReady) return wasmReady;
  const useThreads = canUseWasmThreads();
  const path = useThreads ? WASM_THREADS_PATH : WASM_PATH;
  const label = useThreads ? "threads" : "single";
  if (!existsSync(path)) {
    throw new Error(
      `fail-closed: prebuilt wasm/${label === "threads" ? "milton_threads_bg.wasm" : "milton_bg.wasm"} is missing — this package ships it; do not compile at install`,
    );
  }
  const bytes = readFileSync(path);
  wasmReady = (async () => {
    if (useThreads) {
      const m = await import("../wasm/milton_threads.js");
      const module = new WebAssembly.Module(bytes);
      await m.default({ module_or_path: module });
      const memory = m.wasmMemory();
      const n = resolveThreadCount();
      workerPool = await startWorkerPool({
        module,
        memory,
        workerCount: n,
        miltonSetWorkers: m.miltonSetWorkers,
      });
      lastWasmArtifact = "threads";
      lastThreadCount = workerPool.workerCount;
      api = m;
    } else {
      const m = await import("../wasm/milton.js");
      await m.default({ module_or_path: bytes });
      lastWasmArtifact = "single";
      lastThreadCount = 1;
      api = m;
    }
    lastQ4kCalibration = applyQ4kPolicy(
      {
        q4kSetForce: api.q4kSetForce,
        q4kSetThreshold: api.q4kSetThreshold,
        q4kRunPerk: api.q4kRunPerk,
        q4kRunBprime: api.q4kRunBprime,
        q4kThreshold: api.q4kThreshold,
      },
      process.env,
    );
    return api;
  })();
  return wasmReady;
}

/**
 * Load (or reload) the GGUF into the WASM embedder. Called lazily by `embed`.
 * @param {string} [ggufPath]
 */
export async function load(ggufPath) {
  await ensureWasm();
  const path = ggufPath || resolveGguf();
  if (!existsSync(path)) {
    throw new Error(
      `fail-closed: GGUF not found at ${path} — set MILTON_GGUF or run npm run harness:setup`,
    );
  }
  const bytes = readFileSync(path);
  instance = new api.Milton(bytes);
  return instance;
}

async function ensureModel() {
  if (instance) return instance;
  return load();
}

/**
 * @param {string} text
 * @param {string | { prefix: string, fault?: string }} prefix
 * @returns {Promise<Float32Array>}
 */
export async function embed(text, prefix) {
  if (typeof text !== "string") {
    throw new Error("fail-closed: embed(text, prefix) requires text: string");
  }
  const kind = resolveKind(prefix);
  if (!PREFIX_KINDS.has(kind)) {
    throw new Error(
      `fail-closed: invalid prefix ${JSON.stringify(kind)} (expected 'document' | 'query' | 'none')`,
    );
  }
  const fault = resolveFault(prefix);
  const run = async () => {
    const model = await ensureModel();
    const vec = fault ? model.embedWithFault(text, kind, fault) : model.embed(text, kind);
    return vec instanceof Float32Array ? vec : Float32Array.from(vec);
  };
  const p = queue.then(run, run);
  queue = p.then(
    () => {},
    () => {},
  );
  return p;
}
