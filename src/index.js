/**
 * Public API: `embed(text, prefix) -> Float32Array`.
 *
 * Native Rust forward (this slice, Refs #5). WASM packaging is issue #6.
 * Prefix is config (`document` | `query` | `none`); templates are
 * `search_document: ` / `search_query: ` / passthrough (load-bearing space).
 *
 * Fail-closed: missing milton-embed binary, missing GGUF, or an unverified
 * path refuses. No llama.cpp / onnxruntime in this package — those stay in
 * harness/ as the oracle.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const PREFIX_KINDS = new Set(["document", "query", "none"]);

export function resolveEmbedBin(env = process.env) {
  if (env.MILTON_EMBED_BIN) return env.MILTON_EMBED_BIN;
  const release = join(ROOT, "crate", "target", "release", "milton-embed");
  const debug = join(ROOT, "crate", "target", "debug", "milton-embed");
  if (existsSync(release)) return release;
  if (existsSync(debug)) return debug;
  return null;
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

let child = null;
let buf = "";
/** @type {{ resolve: (line: string) => void, reject: (err: Error) => void }[]} */
let waiters = [];
let queue = Promise.resolve();

function ensureChild() {
  const bin = resolveEmbedBin();
  const gguf = resolveGguf();
  if (!bin) {
    throw new Error(
      "fail-closed: milton-embed binary is missing — build crate/ with `cargo build --release --bin milton-embed` (WASM packaging is issue #6; src/ does not guess)",
    );
  }
  if (!existsSync(gguf)) {
    throw new Error(
      `fail-closed: GGUF not found at ${gguf} — set MILTON_GGUF or run npm run harness:setup`,
    );
  }
  if (child && !child.killed) return child;

  buf = "";
  waiters = [];
  child = spawn(bin, ["--gguf", gguf, "--jsonl"], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    buf += chunk;
    let nl;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const w = waiters.shift();
      if (w) w.resolve(line);
    }
  });
  const fail = (err) => {
    const pending = waiters;
    waiters = [];
    child = null;
    for (const w of pending) w.reject(err);
  };
  child.on("error", (err) => {
    fail(new Error(`fail-closed: milton-embed spawn failed: ${err.message}`));
  });
  child.on("exit", (code, signal) => {
    if (waiters.length) {
      fail(
        new Error(
          `fail-closed: milton-embed exited (code=${code} signal=${signal})`,
        ),
      );
    } else {
      child = null;
    }
  });
  return child;
}

function requestLine(payload) {
  return new Promise((resolve, reject) => {
    const proc = ensureChild();
    waiters.push({ resolve, reject });
    proc.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
      if (err) {
        const w = waiters.pop();
        if (w) w.reject(new Error(`fail-closed: milton-embed stdin: ${err.message}`));
      }
    });
  });
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
  const run = () =>
    requestLine(fault ? { text, prefix: kind, fault } : { text, prefix: kind }).then((line) => {
      let parsed;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new Error(`fail-closed: milton-embed returned non-JSON: ${line.slice(0, 200)}`);
      }
      if (parsed && parsed.error) {
        throw new Error(`fail-closed: ${parsed.error}`);
      }
      if (!parsed || !Array.isArray(parsed.vector) || parsed.vector.length === 0) {
        throw new Error("fail-closed: milton-embed returned no vector");
      }
      return Float32Array.from(parsed.vector);
    });

  const p = queue.then(run, run);
  queue = p.then(
    () => {},
    () => {},
  );
  return p;
}
