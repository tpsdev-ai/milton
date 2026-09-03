/**
 * worker_threads entry for wasm/milton_threads{_relaxed}_bg.wasm.
 * Instantiates the shared-memory module against the coordinator's Memory
 * and parks in `miltonWorkerEnter`. Does not allocate Q8_K or weights.
 */

import { parentPort, workerData } from "node:worker_threads";

const { module, memory, id, threadStackSize, workerGlue } = workerData;
if (module == null || memory == null || id == null || workerGlue == null) {
  throw new Error("fail-closed: wasm worker missing module/memory/id/workerGlue");
}

const { default: init, miltonWorkerEnter } = await import(workerGlue);

await init({
  module_or_path: module,
  memory,
  thread_stack_size: threadStackSize,
});
parentPort.postMessage({ ready: true, id });
miltonWorkerEnter(id);
