#!/usr/bin/env node
/**
 * Baseline bench of the CURRENT Flair embed path on this host.
 *
 * Attempts, in order:
 *   1. harper-fabric-embeddings raw API (the engine Flair registers as
 *      Harper models.embed — same GGUF, mean pool, L2, nomic prefixes).
 *   2. Harper `models.embed` facade (full Flair boot). Unlikely in this
 *      environment; recorded as skipped with the error if it fails.
 *
 * If (1) cannot run, this script exits BLOCKED and writes the attempt log.
 * It does not invent numbers and does not substitute llama-cli as "Flair".
 */
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { loadCorpus } from "../lib/corpus.js";
import { applyPrefix } from "../lib/prefix.js";
import { resolvePaths } from "../lib/reference.js";

const HERE = dirname(fileURLToPath(import.meta.url));

const attempts = [];
const corpus = loadCorpus();
const texts = corpus.cases.map((c) => applyPrefix(c.text, c.prefix));
const paths = resolvePaths();

async function importHfe() {
  const candidates = [
    process.env.MILTON_HFE_MODULE,
    "harper-fabric-embeddings",
    "/tmp/milton-hfe-bench/node_modules/harper-fabric-embeddings/dist/index.js",
    join(HERE, "..", "vendor", "hfe", "node_modules", "harper-fabric-embeddings", "dist", "index.js"),
  ].filter(Boolean);
  const errors = [];
  for (const spec of candidates) {
    try {
      return { mod: await import(spec), spec };
    } catch (e) {
      errors.push(`${spec}: ${e.message}`);
    }
  }
  throw new Error(`import failed:\n${errors.join("\n")}`);
}

async function tryHfe() {
  const attempt = { path: "harper-fabric-embeddings", ok: false };
  try {
    let hfe;
    try {
      const loaded = await importHfe();
      hfe = loaded.mod;
      attempt.resolved = loaded.spec;
    } catch (e) {
      attempt.error = e.message;
      attempts.push(attempt);
      return null;
    }
    if (!existsSync(paths.gguf)) {
      attempt.error = `GGUF missing at ${paths.gguf}`;
      attempts.push(attempt);
      return null;
    }
    // Isolate from a harness llama.cpp build on LD_LIBRARY_PATH — mixing
    // libggml-base.so from a different llama.cpp tree makes HFE's addon
    // report "no backends are loaded".
    const addonPath =
      process.env.MILTON_HFE_ADDON ||
      "/tmp/milton-hfe-bench/node_modules/@node-llama-cpp/linux-x64/bins/linux-x64/llama-addon.node";
    if (existsSync(addonPath)) {
      const addonDir = dirname(addonPath);
      process.env.LD_LIBRARY_PATH = addonDir;
      attempt.addonPath = addonPath;
    }
    const t0 = performance.now();
    await hfe.init({
      modelPath: paths.gguf,
      pooling: "mean",
      threads: 1,
      contextSize: 2048,
      batchSize: 2048,
      ...(existsSync(addonPath) ? { addonPath } : {}),
      templates: {
        document: "search_document: {text}",
        query: "search_query: {text}",
      },
    });
    const first = await hfe.embed(texts[0]);
    const coldMs = performance.now() - t0;
    attempt.cold_start_ms = coldMs;
    attempt.dims = first.length;

    const singleN = Math.min(8, texts.length);
    const t1 = performance.now();
    for (let i = 0; i < singleN; i++) await hfe.embed(texts[i]);
    const singleMs = performance.now() - t1;
    attempt.single = {
      n: singleN,
      ms: singleMs,
      embeddings_per_sec: (singleN / singleMs) * 1000,
    };

    const t2 = performance.now();
    const batch = await hfe.embedBatch(texts);
    const batchMs = performance.now() - t2;
    attempt.batched = {
      n: batch.length,
      ms: batchMs,
      embeddings_per_sec: (batch.length / batchMs) * 1000,
    };

    await hfe.dispose();
    attempt.ok = true;
    attempts.push(attempt);
    return attempt;
  } catch (err) {
    attempt.error = err?.message ?? String(err);
    attempts.push(attempt);
    return null;
  }
}

async function tryHarper() {
  const attempt = { path: "harper.models.embed", ok: false };
  try {
    await import("harper");
    attempt.error =
      "harper imported but no instance boot (models.embed requires a running Harper process) — not measured";
    attempts.push(attempt);
    return null;
  } catch (err) {
    attempt.error = `harper not available: ${err.message}`;
    attempts.push(attempt);
    return null;
  }
}

const os = await import("node:os");
const report = {
  schema: "milton.bench-baseline/1",
  host: {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    cpus: os.availableParallelism(),
  },
  intended_path: "Harper models.embed / harper-fabric-embeddings → llama.cpp",
  attempts,
  status: "pending",
};

const hfe = await tryHfe();
await tryHarper();

if (hfe?.ok) {
  report.status = "ok";
  report.measured = "harper-fabric-embeddings raw API (same engine Flair registers; Harper facade not booted)";
  report.cold_start_ms = hfe.cold_start_ms;
  report.single = hfe.single;
  report.batched = hfe.batched;
  report.dims = hfe.dims;
} else {
  report.status = "BLOCKED";
  report.measured = null;
  report.blocked_reason =
    "Current Flair embed path could not be run in this environment. Numbers were not invented. See attempts.";
}

report.attempts = attempts;

mkdirSync(join(HERE, "..", "receipts"), { recursive: true });
writeFileSync(join(HERE, "..", "receipts", "bench-baseline.json"), `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(JSON.stringify(report, null, 2) + "\n");

if (report.status === "BLOCKED") process.exitCode = 2;
