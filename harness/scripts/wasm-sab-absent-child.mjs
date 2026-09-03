/**
 * Child of wasm-sab-absent.mjs. Runs in a fresh isolate after SAB is gone.
 */
delete globalThis.SharedArrayBuffer;

const milton = await import("../../src/index.js");

if (milton.canUseWasmThreads()) {
  console.error("fail-closed: canUseWasmThreads() is true after deleting SharedArrayBuffer");
  process.exit(1);
}

const vec = await milton.embed("hello", { prefix: "document" });
if (!(vec instanceof Float32Array) || vec.length !== 768) {
  console.error(`fail-closed: sab-absent embed returned ${vec && vec.length}`);
  process.exit(1);
}
if (milton.lastWasmArtifact !== "single") {
  console.error(`fail-closed: loader picked ${milton.lastWasmArtifact}, want single`);
  process.exit(1);
}
if (milton.lastThreadCount !== 1) {
  console.error(`fail-closed: lastThreadCount=${milton.lastThreadCount}, want 1`);
  process.exit(1);
}

process.stdout.write(
  JSON.stringify({
    result: "pass",
    artifact: milton.lastWasmArtifact,
    threads: milton.lastThreadCount,
    dims: vec.length,
    note: "SAB absent is the ordinary path; single-thread module loaded; not an error",
  }) + "\n",
);
