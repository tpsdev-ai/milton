/**
 * Capability probe + worker_threads pool for the shared-memory artifact.
 *
 * Absence of SharedArrayBuffer is the ordinary path (caller loads
 * wasm/milton_bg.wasm). No fallback avalanche, no error.
 */

import { availableParallelism } from "node:os";
import { Worker } from "node:worker_threads";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = join(HERE, "wasm-worker.js");

/** (module (import "e" "m" (memory 1 1 shared))) — section size 9. */
const SHARED_MEMORY_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x02, 0x09, 0x01, 0x01,
  0x65, 0x01, 0x6d, 0x02, 0x03, 0x01, 0x01,
]);

const THREAD_STACK = 2 * 1024 * 1024;

/** Host core count. `1` if `os.availableParallelism` is missing or ≤0. */
export function hostParallelism() {
  try {
    const n = availableParallelism();
    return n > 0 ? n : 1;
  } catch {
    return 1;
  }
}

/**
 * Capability only — not a loader decision. True when SharedArrayBuffer,
 * Atomics, and a shared-memory WebAssembly.validate probe all succeed.
 */
export function sabAvailable(global = globalThis) {
  if (typeof global.SharedArrayBuffer !== "function") return false;
  if (typeof global.Atomics !== "object" || global.Atomics == null) return false;
  if (typeof global.WebAssembly?.validate !== "function") return false;
  try {
    return global.WebAssembly.validate(SHARED_MEMORY_PROBE);
  } catch {
    return false;
  }
}

/**
 * Pick the threaded artifact only when SAB is usable AND the pool would
 * be larger than 1. `MILTON_THREADS=1` (and `MILTON_WASM_THREADS=0`)
 * force `wasm/milton_bg.wasm` — Flint #50 ASK so M4 forced-single vs
 * main is the single-thread module, not threads-with-W=1.
 */
export function canUseWasmThreads(env = process.env, global = globalThis) {
  if (env.MILTON_WASM_THREADS === "0") return false;
  if (resolveThreadCount(env) <= 1) return false;
  return sabAvailable(global);
}

let clampWarned = false;

function warnThreadClamp(raw, applied) {
  if (clampWarned) return;
  clampWarned = true;
  console.warn(`MILTON_THREADS=${raw} is out of range; using ${applied}`);
}

/** `min(MILTON_THREADS || 4, os.availableParallelism())`, at least 1. */
export function resolveThreadCount(env = process.env) {
  const cores = hostParallelism();
  const raw = env.MILTON_THREADS;
  if (raw === undefined || raw === "") {
    return Math.min(4, cores);
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 1) {
    warnThreadClamp(raw, 1);
    return 1;
  }
  const want = Math.floor(n);
  if (want > cores) {
    warnThreadClamp(raw, cores);
    return cores;
  }
  return want;
}

/**
 * Spawn W-1 workers that share `memory` and enter `miltonWorkerEnter`.
 * Worker 0 is the coordinator (this thread). Pool is one-per-module-instance.
 */
export async function startWorkerPool({
  module,
  memory,
  workerCount,
  miltonSetWorkers,
  workerGlue = join(HERE, "..", "wasm", "milton_threads.js"),
}) {
  const w = Math.max(1, workerCount | 0);
  if (w <= 1) {
    miltonSetWorkers(1);
    return { workers: [], workerCount: 1 };
  }
  const workers = [];
  const ready = [];
  for (let id = 1; id < w; id += 1) {
    const worker = new Worker(WORKER_PATH, {
      workerData: {
        module,
        memory,
        id,
        threadStackSize: THREAD_STACK,
        workerGlue,
      },
    });
    ready.push(
      new Promise((resolve, reject) => {
        worker.once("message", resolve);
        worker.once("error", reject);
        worker.once("exit", (code) => {
          if (code !== 0) reject(new Error(`fail-closed: wasm worker ${id} exited ${code}`));
        });
      }),
    );
    workers.push(worker);
  }
  await Promise.all(ready);
  // After worker instantiate: rustc +atomics uses passive segments, but
  // set W only once every instance has started so a start-time data
  // apply cannot reset WORKERS to 1 (silent serial fallback).
  miltonSetWorkers(w);
  // unref after ready: unref-before-ready lets Node exit while we still
  // await the first message (unsettled TLA / CI hang-or-drop). After
  // ready, parked workers must not keep the process alive.
  for (const worker of workers) worker.unref();
  return { workers, workerCount: w };
}
