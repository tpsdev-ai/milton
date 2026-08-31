/**
 * llama.cpp reference embedder (harness-only).
 *
 * Invokes the pinned `llama-embedding` binary with mean pooling and L2
 * normalize. Prefixes are applied in JS via `applyPrefix` so the bytes
 * reaching llama.cpp match Flair/HFE (`search_document: ` / `search_query: `).
 */

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { applyPrefix } from "./prefix.js";
import { l2Normalize } from "./metrics.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR = join(HERE, "..", "vendor");

export const DEFAULT_GGUF = join(VENDOR, "models", "nomic-embed-text-v1.5.Q4_K_M.gguf");
export const DEFAULT_LLAMA_EMBEDDING = join(VENDOR, "llama.cpp", "build", "bin", "llama-embedding");
export const PINNED_GGUF_SHA256 = "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac";

export function resolvePaths(env = process.env) {
  return {
    gguf: env.MILTON_REFERENCE_GGUF || DEFAULT_GGUF,
    bin: env.MILTON_REFERENCE_LLAMA_EMBEDDING || DEFAULT_LLAMA_EMBEDDING,
    llamaDir: env.MILTON_REFERENCE_LLAMACPP || join(VENDOR, "llama.cpp"),
  };
}

function parseEmbeddingOutput(stdout) {
  const text = stdout.toString("utf8");
  // --embd-output-format json (array or OpenAI-shaped)
  const jsonStart = text.indexOf("[");
  const jsonBrace = text.indexOf("{");
  if (jsonBrace !== -1 && (jsonStart === -1 || jsonBrace < jsonStart)) {
    const blob = text.slice(jsonBrace);
    try {
      const parsed = JSON.parse(blob);
      const arr =
        parsed?.data?.[0]?.embedding ??
        parsed?.embedding ??
        (Array.isArray(parsed) ? parsed : null);
      if (Array.isArray(arr) && arr.every((n) => typeof n === "number")) {
        return Float32Array.from(arr);
      }
    } catch {
      // fall through
    }
  }
  if (jsonStart !== -1) {
    const last = text.lastIndexOf("]");
    if (last > jsonStart) {
      try {
        const arr = JSON.parse(text.slice(jsonStart, last + 1));
        if (Array.isArray(arr) && arr.length && typeof arr[0] === "number") {
          return Float32Array.from(arr);
        }
        if (Array.isArray(arr?.[0]) && typeof arr[0][0] === "number") {
          return Float32Array.from(arr[0]);
        }
      } catch {
        // fall through
      }
    }
  }
  // Fallback: last run of floats on a line
  const lines = text.trim().split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const nums = lines[i]
      .trim()
      .split(/[\s,]+/)
      .filter((t) => t.length && /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(t))
      .map(Number);
    if (nums.length >= 32) return Float32Array.from(nums);
  }
  throw new Error(`llama-embedding: could not parse embedding from output:\n${text.slice(-800)}`);
}

export function runLlamaEmbedding(prefixed, paths, extraArgs = []) {
  return new Promise((resolve, reject) => {
    if (!existsSync(paths.bin)) {
      reject(new Error(`llama-embedding not found at ${paths.bin} — run npm run harness:setup`));
      return;
    }
    if (!existsSync(paths.gguf)) {
      reject(new Error(`GGUF not found at ${paths.gguf} — run npm run harness:setup`));
      return;
    }
    const promptDir = join(tmpdir(), "milton-harness");
    mkdirSync(promptDir, { recursive: true });
    const promptFile = join(promptDir, `prompt-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}.txt`);
    writeFileSync(promptFile, prefixed, "utf8");
    const args = [
      "-m",
      paths.gguf,
      "-f",
      promptFile,
      "--no-escape",
      "--pooling",
      "mean",
      "--embd-normalize",
      "2",
      "--embd-output-format",
      "json",
      "-t",
      "1",
      "-c",
      "2048",
      "-b",
      "2048",
      "-ub",
      "2048",
      "--no-warmup",
      ...extraArgs,
    ];
    const binDir = dirname(paths.bin);
    const child = spawn(paths.bin, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, LD_LIBRARY_PATH: `${binDir}${process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : ""}` },
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (d) => stdout.push(d));
    child.stderr.on("data", (d) => stderr.push(d));
    child.on("error", reject);
    child.on("close", (code) => {
      try { unlinkSync(promptFile); } catch { /* tmp */ }
      const out = Buffer.concat(stdout);
      const err = Buffer.concat(stderr).toString("utf8");
      if (code !== 0) {
        reject(new Error(`llama-embedding exited ${code}: ${err.slice(-1200)}`));
        return;
      }
      try {
        let vec = parseEmbeddingOutput(out.length ? out : Buffer.from(err));
        // Belt: HFE L2-normalizes; --embd-normalize 2 should already have.
        vec = l2Normalize(vec);
        resolve(vec);
      } catch (e) {
        reject(new Error(`${e.message}\nstderr:\n${err.slice(-800)}`));
      }
    });
  });
}

export function createReferenceEmbedder(paths = resolvePaths(), extraArgs = []) {
  return async function embed(text, { prefix = "none" } = {}) {
    const prefixed = applyPrefix(text, prefix);
    return runLlamaEmbedding(prefixed, paths, extraArgs);
  };
}

export function readLlamaCppPin(llamaDir) {
  const head = join(llamaDir, ".git", "HEAD");
  if (!existsSync(head)) return { commit: null, describe: null };
  let ref = readFileSync(head, "utf8").trim();
  if (ref.startsWith("ref: ")) {
    const refPath = join(llamaDir, ".git", ref.slice(5));
    ref = existsSync(refPath) ? readFileSync(refPath, "utf8").trim() : null;
  }
  return { commit: ref, describe: ref };
}
