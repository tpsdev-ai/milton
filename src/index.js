/**
 * Public API: `embed(text, {prefix}) -> Float32Array`.
 *
 * WASM-SIMD path (Refs #6). Same Rust `Model` as the native bins, compiled
 * with +simd128. Prefix is config (`document` | `query` | `none`); templates
 * are `search_document: ` / `search_query: ` / passthrough (load-bearing space).
 *
 * Fail-closed: missing prebuilt wasm, missing GGUF, or an unverified path
 * refuses. No native compile, no node-gyp, no per-platform build at install.
 * The reference toolchain stays in harness/ as the oracle.
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import init, { Milton } from "../wasm/milton.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const WASM_PATH = join(ROOT, "wasm", "milton_bg.wasm");

const PREFIX_KINDS = new Set(["document", "query", "none"]);

export function resolveWasmPath() {
  return WASM_PATH;
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

function ensureWasm() {
  if (wasmReady) return wasmReady;
  if (!existsSync(WASM_PATH)) {
    throw new Error(
      "fail-closed: prebuilt wasm/milton_bg.wasm is missing — this package ships it; do not compile at install",
    );
  }
  const bytes = readFileSync(WASM_PATH);
  wasmReady = init({ module_or_path: bytes });
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
  instance = new Milton(bytes);
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
