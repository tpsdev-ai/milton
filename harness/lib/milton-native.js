/**
 * Native milton-embed JSONL client. Harness-only — not the shipped API.
 * Used to compare WASM-SIMD against the same crate compiled native.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

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

/**
 * @returns {(text: string, opts: { prefix: string }) => Promise<Float32Array>}
 */
export function createNativeEmbedder() {
  const bin = resolveEmbedBin();
  const gguf = resolveGguf();
  if (!bin) {
    throw new Error(
      "fail-closed: milton-embed binary is missing — build crate/ with `cargo build --release --bin milton-embed`",
    );
  }
  if (!existsSync(gguf)) {
    throw new Error(`fail-closed: GGUF not found at ${gguf}`);
  }

  let child = null;
  let buf = "";
  /** @type {{ resolve: (line: string) => void, reject: (err: Error) => void }[]} */
  let waiters = [];
  let queue = Promise.resolve();

  function ensureChild() {
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
        fail(new Error(`fail-closed: milton-embed exited (code=${code} signal=${signal})`));
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

  return async function embed(text, prefix) {
    const kind = typeof prefix === "string" ? prefix : prefix?.prefix;
    const p = queue.then(() =>
      requestLine({ text, prefix: kind }).then((line) => {
        const parsed = JSON.parse(line);
        if (parsed.error) throw new Error(`fail-closed: ${parsed.error}`);
        if (!parsed.vector) throw new Error("fail-closed: milton-embed returned no vector");
        return Float32Array.from(parsed.vector);
      }),
    );
    queue = p.then(
      () => {},
      () => {},
    );
    return p;
  };
}
