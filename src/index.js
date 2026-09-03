/**
 * Public API: `embed(text, {prefix}) -> Float32Array`.
 *
 * WASM-SIMD path (Refs #6). Same Rust `Model` as the native bins, compiled
 * with +simd128. Prefix is config (`document` | `query` | `none`); templates
 * are `search_document: ` / `search_query: ` / passthrough (load-bearing space).
 *
 * Two thread artifacts, one loader (Refs #44). The shared-memory module is
 * used only when SharedArrayBuffer + Atomics exist, WebAssembly.validate
 * accepts a shared-memory probe, and the pool would be larger than 1.
 * `MILTON_THREADS=1` forces the single-thread module (not threads-with-W=1).
 * Absence of SAB is the ordinary path — not an error.
 *
 * Single-thread pick (Refs #43): `WebAssembly.validate` of a relaxed-dot
 * probe loads `wasm/milton_relaxed_bg.wasm`; otherwise `milton_bg.wasm`.
 * Threaded pick: when SAB + pool > 1, load `milton_threads_relaxed_bg.wasm`
 * when the relaxed probe passes, else `milton_threads_bg.wasm`.
 * `MILTON_RELAXED_SIMD=0` forces simd128 on both paths; `=1` fail-closes
 * if the probe rejects. Never a Node version sniff — bun validates SIMD128
 * and rejects the relaxed probe.
 *
 * Fail-closed: missing prebuilt wasm, missing GGUF, or an unverified path
 * refuses. No native compile and no per-platform build at install.
 * The reference toolchain stays in harness/ as the oracle.
 */

import { readFileSync, existsSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { applyQ4kPolicy } from "./q4k-calibrate.js";
import { resolveQmatmulKernel } from "./relaxed-simd.js";
import {
  canUseWasmThreads,
  hostParallelism,
  resolveThreadCount,
  sabAvailable,
  startWorkerPool,
} from "./wasm-threads.js";

export { probeRelaxedSimd, resolveQmatmulKernel } from "./relaxed-simd.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const WASM_PATH = join(ROOT, "wasm", "milton_bg.wasm");
const WASM_RELAXED_PATH = join(ROOT, "wasm", "milton_relaxed_bg.wasm");
const WASM_THREADS_PATH = join(ROOT, "wasm", "milton_threads_bg.wasm");
const WASM_THREADS_RELAXED_PATH = join(ROOT, "wasm", "milton_threads_relaxed_bg.wasm");
const GLUE_THREADS = "../wasm/milton_threads.js";
const GLUE_THREADS_RELAXED = "../wasm/milton_threads_relaxed.js";

const PREFIX_KINDS = new Set(["document", "query", "none"]);

export function resolveWasmPath() {
  return WASM_PATH;
}

export function resolveThreadsWasmPath(relaxed = false) {
  return relaxed ? WASM_THREADS_RELAXED_PATH : WASM_THREADS_PATH;
}

export function resolveRelaxedWasmPath() {
  return WASM_RELAXED_PATH;
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

/** Basename of the `.wasm` file this process actually instantiated. */
export let lastWasmFile = null;

/**
 * Loader path report (Flint #50 ASK). Same grain as `lastQ4kCalibration`.
 * `sabAvailable` is the capability probe, not "did we pick threads" —
 * `MILTON_THREADS=1` on a SAB host is `{artifact:'single', workers:1, sabAvailable:true}`.
 * `wasm` is the basename actually instantiated (proves the four-way pick).
 * On a failed load, the same grain is published with `error` (string) and
 * `wasm` set to the artifact that was attempted — never a prior success.
 * @type {{ artifact: 'single' | 'threads', workers: number, availableParallelism: number, sabAvailable: boolean, wasm: string, error?: string } | null}
 */
export let lastThreadReport = null;

/** `'single' | 'threads'` — alias of `lastThreadReport.artifact`. */
export let lastWasmArtifact = null;

/** Pool size after load. `1` on the single-thread artifact. */
export let lastThreadCount = 1;

/**
 * Q4_K/Q5_K integer-tree pick (issue #43). Same grain as `lastThreadReport`.
 * `probe` is the capability (`WebAssembly.validate` of relaxed-dot), not the pick.
 * Applies on both single-thread and threaded artifacts.
 * On a failed load: `{error, wasm}` plus kernel/probe/forced when the pick
 * completed before the failure. Never a prior success without `error`.
 * @type {{ kernel?: 'relaxed' | 'simd128', probe?: boolean, forced?: boolean, error?: string, wasm?: string } | null}
 */
export let lastQmatmulKernel = null;

export { canUseWasmThreads, hostParallelism, resolveThreadCount, sabAvailable };

function publishThreadReport(artifact, workers, wasmFile) {
  lastWasmFile = wasmFile;
  lastThreadReport = {
    artifact,
    workers,
    availableParallelism: hostParallelism(),
    sabAvailable: sabAvailable(),
    wasm: wasmFile,
  };
  lastWasmArtifact = artifact;
  lastThreadCount = workers;
}

function errorText(err) {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Every load-failure path publishes both reports so a consumer cannot
 * read a prior success (or null) after this attempt failed. Refs #54.
 * `wasmFile` is the basename/path the loader was trying.
 */
function publishLoadError(err, { artifact, workers, wasmFile, kernel } = {}) {
  const error = errorText(err);
  const wasm = wasmFile || lastWasmFile || "";
  const art =
    artifact === "threads" || artifact === "single"
      ? artifact
      : lastThreadReport?.artifact === "threads"
        ? "threads"
        : "single";
  const w = Number.isFinite(workers) ? workers : lastThreadCount;
  lastWasmFile = wasm || lastWasmFile;
  lastThreadReport = {
    artifact: art,
    workers: w,
    availableParallelism: hostParallelism(),
    sabAvailable: sabAvailable(),
    wasm,
    error,
  };
  lastWasmArtifact = art;
  lastThreadCount = w;
  const pick =
    kernel && typeof kernel.kernel === "string"
      ? { kernel: kernel.kernel, probe: Boolean(kernel.probe), forced: Boolean(kernel.forced) }
      : {};
  lastQmatmulKernel = { ...pick, error, wasm };
}

function ensureWasm() {
  if (wasmReady) return wasmReady;
  let artifact = "single";
  let workers = 1;
  let missingName = "milton_relaxed_bg.wasm";
  let path;
  let glueModule;
  let qk;
  let useThreads;
  try {
    useThreads = canUseWasmThreads();
    if (useThreads) {
      artifact = "threads";
      workers = resolveThreadCount();
      missingName = "milton_threads_relaxed_bg.wasm";
    }
    qk = resolveQmatmulKernel(process.env);
    const useRelaxedKernel = qk.kernel === "relaxed";
    if (useThreads) {
      path = useRelaxedKernel ? WASM_THREADS_RELAXED_PATH : WASM_THREADS_PATH;
      missingName = useRelaxedKernel
        ? "milton_threads_relaxed_bg.wasm"
        : "milton_threads_bg.wasm";
      glueModule = useRelaxedKernel ? GLUE_THREADS_RELAXED : GLUE_THREADS;
    } else {
      path = useRelaxedKernel ? WASM_RELAXED_PATH : WASM_PATH;
      missingName = useRelaxedKernel ? "milton_relaxed_bg.wasm" : "milton_bg.wasm";
      glueModule = useRelaxedKernel ? "../wasm/milton_relaxed.js" : "../wasm/milton.js";
    }
    if (!existsSync(path)) {
      throw new Error(
        `fail-closed: prebuilt wasm/${missingName} is missing — this package ships it; do not compile at install`,
      );
    }
    const bytes = readFileSync(path);
    wasmReady = (async () => {
      try {
        if (useThreads) {
          const m = await import(glueModule);
          const module = new WebAssembly.Module(bytes);
          await m.default({ module_or_path: module });
          const memory = m.wasmMemory();
          const n = resolveThreadCount();
          workerPool = await startWorkerPool({
            module,
            memory,
            workerCount: n,
            miltonSetWorkers: m.miltonSetWorkers,
            workerGlue: glueModule,
          });
          publishThreadReport("threads", workerPool.workerCount, missingName);
          lastQmatmulKernel = qk;
          api = m;
        } else {
          const m = await import(glueModule);
          await m.default({ module_or_path: bytes });
          publishThreadReport("single", 1, missingName);
          lastQmatmulKernel = qk;
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
      } catch (err) {
        publishLoadError(err, { artifact, workers, wasmFile: missingName, kernel: qk });
        throw err;
      }
    })();
    return wasmReady;
  } catch (err) {
    publishLoadError(err, { artifact, workers, wasmFile: missingName, kernel: qk });
    throw err;
  }
}

/**
 * Load (or reload) the GGUF into the WASM embedder. Called lazily by `embed`.
 * @param {string} [ggufPath]
 */
export async function load(ggufPath) {
  await ensureWasm();
  const path = ggufPath || resolveGguf();
  const attempted = basename(path);
  if (!existsSync(path)) {
    const err = new Error(
      `fail-closed: GGUF not found at ${path} — set MILTON_GGUF or run npm run harness:setup`,
    );
    publishLoadError(err, {
      artifact: lastThreadReport?.artifact,
      workers: lastThreadCount,
      wasmFile: attempted,
      kernel: lastQmatmulKernel,
    });
    throw err;
  }
  try {
    const bytes = readFileSync(path);
    instance = new api.Milton(bytes);
    return instance;
  } catch (err) {
    publishLoadError(err, {
      artifact: lastThreadReport?.artifact,
      workers: lastThreadCount,
      wasmFile: attempted,
      kernel: lastQmatmulKernel,
    });
    throw err;
  }
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
