#!/usr/bin/env node
/**
 * Issue #56: product-path stage tables at 1 and 4 workers, short and long-n.
 * Builds a separate threads+relaxed+profile wasm — never overwrites wasm/.
 */
import { spawnSync, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";
import { loadCorpus } from "../lib/corpus.js";
import { resolveGguf } from "../../src/index.js";
import { startWorkerPool } from "../../src/wasm-threads.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const CORPUS = loadCorpus();
const GGUF = resolveGguf();
const OUT_DIR = join(HERE, "..", "profile");
const OUT_JSON = join(OUT_DIR, "threads-stage-profile.json");
const OUT_MD = join(OUT_DIR, "threads-stage-profile.tables.md");
const PROFILE_DIR = join(ROOT, "crate", "target", "profile-wasm-threads-relaxed");

const ATTN_STAGES = ["attn_qk", "attn_softmax", "attn_vmix"];
const MATMUL_STAGES = ["qkv", "out_proj", "ffn_gate", "ffn_up", "ffn_down"];
const CASES = ["short-hello-document", "long-repeated"];
const WORKERS = [1, 4];

function fmt(n, d = 2) {
  if (n == null || !Number.isFinite(n)) return "—";
  return Number(n).toFixed(d);
}

function fmtPct(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  return `${n.toFixed(1)}%`;
}

function hostInfo() {
  let cpu = {};
  try {
    const raw = execFileSync("lscpu", { encoding: "utf8" });
    const grab = (re) => (raw.match(re) || [])[1] || null;
    cpu = {
      model: grab(/Model name:\s+(.+)/),
      cpus: Number(grab(/CPU\(s\):\s+(\d+)/)),
      mhz: Number(grab(/CPU MHz:\s+([\d.]+)/)),
    };
  } catch {
    cpu = { error: "lscpu failed" };
  }
  return {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    cpu,
  };
}

function buildProfileThreads() {
  const env = {
    ...process.env,
    MILTON_PROFILE_THREADS: "1",
    MILTON_PROFILE_RELAXED: "1",
  };
  const r = spawnSync("bash", [join(ROOT, "scripts", "build-wasm-profile.sh"), PROFILE_DIR], {
    encoding: "utf8",
    cwd: ROOT,
    env,
  });
  if (r.status !== 0) {
    throw new Error(`build-wasm-profile threads failed:\n${r.stderr}\n${r.stdout}`);
  }
  return PROFILE_DIR;
}

async function loadProfiler(dir, workers) {
  const glue = join(dir, "milton.js");
  const wasmPath = join(dir, "milton_bg.wasm");
  const bytes = readFileSync(wasmPath);
  const m = await import(pathToFileURL(glue).href);
  const module = new WebAssembly.Module(bytes);
  await m.default({ module_or_path: module });
  const memory = m.wasmMemory();
  await startWorkerPool({
    module,
    memory,
    workerCount: workers,
    miltonSetWorkers: m.miltonSetWorkers,
    workerGlue: glue,
  });
  const inst = new m.Milton(readFileSync(GGUF));
  return {
    workers,
    wasmBytes: statSync(wasmPath).size,
    profile(text, prefix) {
      const t0 = performance.now();
      const raw = inst.embedProfiled(text, prefix);
      const jsMs = performance.now() - t0;
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      parsed.js_wall_ms = jsMs;
      return parsed;
    },
  };
}

function summarize(snap) {
  const attn_ms = ATTN_STAGES.reduce((s, n) => s + (snap.stages_ms?.[n] ?? 0), 0);
  const matmul_ms = MATMUL_STAGES.reduce((s, n) => s + (snap.stages_ms?.[n] ?? 0), 0);
  const rope_ms = snap.stages_ms?.rope ?? 0;
  const ln_ms = snap.stages_ms?.layernorm ?? 0;
  const swiglu_ms = snap.stages_ms?.ffn_swiglu ?? 0;
  return {
    total_ms: snap.total_ms,
    matmul_ms,
    attn_ms,
    rope_ms,
    ln_ms,
    swiglu_ms,
    matmul_pct: snap.total_ms > 0 ? (100 * matmul_ms) / snap.total_ms : 0,
    attn_pct: snap.total_ms > 0 ? (100 * attn_ms) / snap.total_ms : 0,
    rope_pct: snap.total_ms > 0 ? (100 * rope_ms) / snap.total_ms : 0,
    ln_pct: snap.total_ms > 0 ? (100 * ln_ms) / snap.total_ms : 0,
    swiglu_pct: snap.total_ms > 0 ? (100 * swiglu_ms) / snap.total_ms : 0,
    stages_ms: snap.stages_ms,
  };
}

async function runOneWorker(dir, w) {
  const prof = await loadProfiler(dir, w);
  const rows = [];
  for (const id of CASES) {
    const c = CORPUS.cases.find((x) => x.id === id);
    if (!c) throw new Error(`missing corpus case ${id}`);
    const repeats = id === "long-repeated" ? 1 : 2;
    console.error(`profile ${id} W=${w} repeats=${repeats}`);
    let mid = null;
    for (let i = 0; i < repeats; i++) {
      const got = prof.profile(c.text, c.prefix);
      if (!mid || got.profile.total_ms < mid.profile.total_ms) mid = got;
    }
    const sum = summarize(mid.profile);
    rows.push({
      id,
      workers: w,
      n_tokens: mid.profile.n_tokens,
      js_wall_ms: mid.js_wall_ms,
      ...sum,
    });
    console.error(
      `  ${id} W=${w} n=${mid.profile.n_tokens} total=${fmt(sum.total_ms)} attn=${fmtPct(sum.attn_pct)} rope=${fmt(sum.rope_ms)} ln=${fmt(sum.ln_ms)} swiglu=${fmt(sum.swiglu_ms)}`,
    );
  }
  return rows;
}

async function main() {
  if (!existsSync(GGUF)) {
    console.error(`fail-closed: GGUF not found at ${GGUF}`);
    process.exit(2);
  }
  const host = hostInfo();
  const childW = process.env.MILTON_PROFILE_W;
  if (childW) {
    const dir = process.env.MILTON_PROFILE_DIR || PROFILE_DIR;
    const rows = await runOneWorker(dir, Number(childW));
    process.stdout.write(`${JSON.stringify(rows)}\n`);
    return;
  }
  console.error("building threads+relaxed+profile wasm...");
  const dir = buildProfileThreads();
  const rows = [];
  for (const w of WORKERS) {
    console.error(`child W=${w}`);
    const r = spawnSync(
      process.execPath,
      [fileURLToPath(import.meta.url)],
      {
        encoding: "utf8",
        cwd: ROOT,
        env: { ...process.env, MILTON_PROFILE_W: String(w), MILTON_PROFILE_DIR: dir },
        maxBuffer: 16 * 1024 * 1024,
      },
    );
    if (r.status !== 0) {
      throw new Error(`profile child W=${w} failed:\n${r.stderr}\n${r.stdout}`);
    }
    const line = r.stdout.trim().split("\n").pop();
    rows.push(...JSON.parse(line));
  }

  const md = [];
  md.push("# #56 product-path stage tables (threads + relaxed profile wasm)");
  md.push("");
  md.push(`Host: ${host.cpu.model || host.arch}, Node ${host.node}, ${host.cpu.cpus} CPUs.`);
  md.push("W=1 is the threads artifact with `pool_live()=false` (serial attention).");
  md.push("W=4 is head-split attention (phase A2). RoPE / LN / SwiGLU stay on the coordinator.");
  md.push("Under W=4, qk+softmax+V-mix wall is recorded as `attn_qk` (sub-stages fused in the join).");
  md.push("");
  md.push("| workers | case | n | total ms | matmul % | attn % | attn ms | RoPE ms | LN ms | SwiGLU ms |");
  md.push("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const r of rows) {
    md.push(
      `| ${r.workers} | ${r.id} | ${r.n_tokens} | ${fmt(r.total_ms)} | ${fmtPct(r.matmul_pct)} | ${fmtPct(r.attn_pct)} | ${fmt(r.attn_ms)} | ${fmt(r.rope_ms, 3)} | ${fmt(r.ln_ms, 3)} | ${fmt(r.swiglu_ms, 3)} |`,
    );
  }
  md.push("");

  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(OUT_JSON, `${JSON.stringify({ schema: "milton.threads-stage-profile/1", issue: 56, host, rows }, null, 2)}\n`);
  writeFileSync(OUT_MD, `${md.join("\n")}\n`);
  process.stdout.write(`${md.join("\n")}\n`);
  process.stdout.write(`\nwrote ${OUT_JSON}\n`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
