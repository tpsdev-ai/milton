/**
 * llama.cpp reference embedder (harness-only).
 *
 * Invokes the pinned `llama-embedding` binary with mean pooling and L2
 * normalize. Prefixes are applied in JS via `applyPrefix` so the bytes
 * reaching llama.cpp match Flair/HFE (`search_document: ` / `search_query: `).
 */

import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { applyPrefix } from "./prefix.js";
import { l2Normalize } from "./metrics.js";
import { TOKEN_ID_ORACLE_SOURCE, TOKEN_ID_ORACLE_TOOL } from "./token-id-oracle.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR = join(HERE, "..", "vendor");

export const DEFAULT_GGUF = join(VENDOR, "models", "nomic-embed-text-v1.5.Q4_K_M.gguf");
export const DEFAULT_LLAMA_EMBEDDING = join(VENDOR, "llama.cpp", "build", "bin", "llama-embedding");
export const DEFAULT_EMBED_FROM_TOKEN_IDS = join(VENDOR, "bin", "embed-from-token-ids");
export const PINNED_GGUF_SHA256 = "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac";
export const BUILD_EMBED_FROM_TOKEN_IDS = join(HERE, "..", "scripts", "build-embed-from-token-ids.sh");

export function resolvePaths(env = process.env) {
  return {
    gguf: env.MILTON_REFERENCE_GGUF || DEFAULT_GGUF,
    bin: env.MILTON_REFERENCE_LLAMA_EMBEDDING || DEFAULT_LLAMA_EMBEDDING,
    embedFromTokenIds: env.MILTON_REFERENCE_EMBED_FROM_TOKEN_IDS || DEFAULT_EMBED_FROM_TOKEN_IDS,
    llamaDir: env.MILTON_REFERENCE_LLAMACPP || join(VENDOR, "llama.cpp"),
  };
}

export function ensureEmbedFromTokenIds(paths = resolvePaths()) {
  if (existsSync(paths.embedFromTokenIds)) return paths.embedFromTokenIds;
  execFileSync("bash", [BUILD_EMBED_FROM_TOKEN_IDS], { stdio: "inherit" });
  if (!existsSync(paths.embedFromTokenIds)) {
    throw new Error(`fail-closed: ${TOKEN_ID_ORACLE_TOOL} did not produce ${paths.embedFromTokenIds}`);
  }
  return paths.embedFromTokenIds;
}

/**
 * llama.cpp GGUF forward on exact token IDs (no text tokenizer).
 * Independent of Milton. Token IDs must come from tokens.json (HF pin).
 */
export function runEmbedFromTokenIds(ids, paths = resolvePaths()) {
  if (!Array.isArray(ids) || !ids.length || !ids.every((n) => Number.isInteger(n))) {
    throw new Error("fail-closed: token IDs must be a non-empty integer array from tokens.json");
  }
  const bin = ensureEmbedFromTokenIds(paths);
  if (!existsSync(paths.gguf)) {
    throw new Error(`GGUF not found at ${paths.gguf} — run npm run harness:setup`);
  }
  const binDir = dirname(paths.bin);
  const stdout = execFileSync(bin, [paths.gguf, "--ids", ids.join(",")], {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    env: {
      ...process.env,
      LD_LIBRARY_PATH: `${binDir}${process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : ""}`,
    },
  });
  const jsonStart = stdout.indexOf("{");
  if (jsonStart < 0) {
    throw new Error(`embed-from-token-ids: no JSON on stdout:\n${stdout.slice(-400)}`);
  }
  const parsed = JSON.parse(stdout.slice(jsonStart));
  if (parsed?.schema !== "milton.embed-from-token-ids/1" || !Array.isArray(parsed.embedding)) {
    throw new Error("fail-closed: embed-from-token-ids returned an unexpected schema");
  }
  if (parsed.n_ids !== ids.length || parsed.ids?.length !== ids.length) {
    throw new Error(
      `fail-closed: embed-from-token-ids n_ids=${parsed.n_ids} != requested ${ids.length}`,
    );
  }
  for (let i = 0; i < ids.length; i++) {
    if (parsed.ids[i] !== ids[i]) {
      throw new Error(`fail-closed: embed-from-token-ids echoed id[${i}]=${parsed.ids[i]} != ${ids[i]}`);
    }
  }
  return {
    vector: l2Normalize(Float32Array.from(parsed.embedding)),
    provenance: {
      oracle: "embed-from-token-ids",
      tool: TOKEN_ID_ORACLE_TOOL,
      token_ids_source: TOKEN_ID_ORACLE_SOURCE,
      n_ids: ids.length,
      ids: [...ids],
      pooling: parsed.pooling,
      embd_normalize: parsed.embd_normalize,
    },
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
