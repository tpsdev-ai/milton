#!/usr/bin/env node
/**
 * Issue #28 measurement chip. Per-stage wall-clock, native AND WASM,
 * same wasm:bench corpus, plus Milton-native vs llama.cpp n_threads=1.
 *
 * Does not overwrite wasm/milton_bg.wasm. Profile artifacts go under
 * crate/target/profile-*. Default gates stay on the unprofiled path.
 */
import { spawn, spawnSync, execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createNativeEmbedder } from "../lib/milton-native.js";
import { loadCorpus } from "../lib/corpus.js";
import { applyPrefix } from "../lib/prefix.js";
import { resolvePaths } from "../lib/reference.js";
import { resolveGguf } from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const CORPUS = loadCorpus();
const TOKENS = JSON.parse(
  readFileSync(join(HERE, "..", "goldens", "tokens.json"), "utf8"),
);
const BASELINE = JSON.parse(
  readFileSync(join(HERE, "..", "goldens", "baseline-bench.json"), "utf8"),
);
const GGUF = resolveGguf();
const OUT_JSON = join(HERE, "..", "goldens", "stage-profile.json");

const STAGE_ORDER = [
  "tokenize",
  "embedding_lookup",
  "qkv",
  "rope",
  "attn_qk",
  "attn_softmax",
  "attn_vmix",
  "out_proj",
  "ffn_gate",
  "ffn_up",
  "ffn_swiglu",
  "ffn_down",
  "layernorm",
  "pooling",
];

const MATMUL_STAGES = ["qkv", "out_proj", "ffn_gate", "ffn_up", "ffn_down"];
const ATTN_STAGES = ["attn_qk", "attn_softmax", "attn_vmix"];

function nIds(id) {
  const item = TOKENS.items.find((x) => x.id === id);
  return item?.n_ids ?? null;
}

function median(xs) {
  const a = [...xs].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
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
      l1d: grab(/L1d cache:\s+(.+)/),
      l2: grab(/L2 cache:\s+(.+)/),
      l3: grab(/L3 cache:\s+(.+)/),
      avx2: /avx2/.test(raw),
      avx512f: /avx512f/.test(raw),
      fma: /\bfma\b/.test(raw),
    };
  } catch {
    cpu = { error: "lscpu failed" };
  }
  const freqGhz = (cpu.mhz && cpu.mhz > 0 ? cpu.mhz : 2400) / 1000;
  // Milton native kernels are AVX2 FMA (2× 8-wide FMA = 32 FLOP/cycle).
  // WASM SIMD128 has no IEEE FMA: 4-wide mul+add, optimistic 8 FLOP/cycle.
  const roofline = {
    freq_ghz: freqGhz,
    avx2_flop_per_cycle: 32,
    simd128_flop_per_cycle: 8,
    avx2_peak_gflops: 32 * freqGhz,
    simd128_peak_gflops: 8 * freqGhz,
    note: "single-thread. AVX2 = 2 FMA ports × 8-wide × 2 flop. SIMD128 = 4-wide mul+add, no FMA.",
  };
  return {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    cpu,
    roofline,
  };
}

function buildNativeProfile() {
  const r = spawnSync("bash", [join(ROOT, "scripts", "build-native-profile.sh")], {
    encoding: "utf8",
    cwd: ROOT,
  });
  if (r.status !== 0) {
    throw new Error(`build-native-profile failed:\n${r.stderr}\n${r.stdout}`);
  }
  const bin = r.stdout.trim().split("\n").pop();
  if (!bin || !existsSync(bin)) throw new Error("milton-profile binary missing");
  return bin;
}

function buildDefaultEmbed() {
  const r = spawnSync(
    "cargo",
    ["build", "--manifest-path", "crate/Cargo.toml", "--release", "--bin", "milton-embed"],
    { encoding: "utf8", cwd: ROOT },
  );
  if (r.status !== 0) {
    throw new Error(`milton-embed build failed:\n${r.stderr}`);
  }
  return join(ROOT, "crate", "target", "release", "milton-embed");
}

function buildProfileWasm() {
  const out = join(ROOT, "crate", "target", "profile-wasm");
  const r = spawnSync("bash", [join(ROOT, "scripts", "build-wasm-profile.sh"), out], {
    encoding: "utf8",
    cwd: ROOT,
  });
  if (r.status !== 0) {
    throw new Error(`build-wasm-profile failed:\n${r.stderr}\n${r.stdout}`);
  }
  return out;
}

function createNativeProfiler(bin) {
  let child = null;
  let buf = "";
  let waiters = [];
  let queue = Promise.resolve();

  function ensure() {
    if (child && !child.killed) return child;
    buf = "";
    waiters = [];
    child = spawn(bin, ["--gguf", GGUF, "--jsonl"], { stdio: ["pipe", "pipe", "pipe"] });
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
    child.on("error", (err) => fail(new Error(`milton-profile spawn: ${err.message}`)));
    child.on("exit", (code, signal) => {
      if (waiters.length) fail(new Error(`milton-profile exited code=${code} signal=${signal}`));
      else child = null;
    });
    return child;
  }

  function request(payload) {
    return new Promise((resolve, reject) => {
      const proc = ensure();
      waiters.push({ resolve, reject });
      proc.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
        if (err) {
          const w = waiters.pop();
          if (w) w.reject(err);
        }
      });
    });
  }

  async function profile(text, prefix) {
    const p = queue.then(async () => {
      const line = await request({ text, prefix });
      const parsed = JSON.parse(line);
      if (parsed.error) throw new Error(parsed.error);
      return parsed;
    });
    queue = p.then(
      () => {},
      () => {},
    );
    return p;
  }

  function close() {
    if (child && !child.killed) {
      child.stdin.end();
      child.kill("SIGTERM");
    }
    child = null;
  }

  return { profile, close };
}

async function createWasmProfiler(dir) {
  const glue = join(dir, "milton.js");
  const wasmPath = join(dir, "milton_bg.wasm");
  const mod = await import(pathToFileURL(glue).href);
  const init = mod.default;
  const Milton = mod.Milton;
  await init({ module_or_path: readFileSync(wasmPath) });
  const inst = new Milton(readFileSync(GGUF));
  return {
    profile(text, prefix) {
      const t0 = performance.now();
      const raw = inst.embedProfiled(text, prefix);
      const jsMs = performance.now() - t0;
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      parsed.js_wall_ms = jsMs;
      return parsed;
    },
    wasmBytes: statSync(wasmPath).size,
  };
}

function vecEqual(a, b) {
  if (a.length !== b.length) return { ok: false, reason: `len ${a.length} vs ${b.length}` };
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return { ok: false, reason: `elem[${i}] ${a[i]} vs ${b[i]}` };
    }
  }
  return { ok: true };
}

function summarizeStages(snap) {
  const stages = {};
  let accounted = 0;
  for (const name of STAGE_ORDER) {
    const ms = snap.stages_ms?.[name] ?? 0;
    stages[name] = { ms, pct: snap.total_ms > 0 ? (100 * ms) / snap.total_ms : 0 };
    accounted += ms;
  }
  const matmul_ms = MATMUL_STAGES.reduce((s, n) => s + (snap.stages_ms?.[n] ?? 0), 0);
  const attn_ms = ATTN_STAGES.reduce((s, n) => s + (snap.stages_ms?.[n] ?? 0), 0);
  return {
    stages,
    matmul_ms,
    attn_ms,
    matmul_pct: snap.total_ms > 0 ? (100 * matmul_ms) / snap.total_ms : 0,
    attn_pct: snap.total_ms > 0 ? (100 * attn_ms) / snap.total_ms : 0,
    accounted_ms: accounted,
    unaccounted_ms: snap.total_ms - accounted,
  };
}

function gflops(flops, ms) {
  if (!(ms > 0) || !(flops > 0)) return null;
  return flops / (ms * 1e6);
}

function stageGflops(snap, name, flops, peak) {
  const ms = snap.stages_ms?.[name] ?? 0;
  const gf = gflops(flops, ms);
  return {
    ms,
    flops,
    gflops: gf,
    pct_of_roofline: gf != null && peak > 0 ? (100 * gf) / peak : null,
  };
}

function attachRoofline(snap, peak) {
  const c = snap.counters || {};
  const n = snap.n_tokens;
  const qkFlops = Number(c.attn_qk_flops || 0);
  const vmixFlops = Number(c.attn_vmix_flops || 0);
  const matmulFlops = Number(c.matmul_flops || 0);
  // Split matmul FLOPs by stage using n_in/n_out of nomic-bert.
  const nEmbd = c.n_embd || 768;
  const nFf = c.n_ff || 3072;
  const nLayer = c.n_layer || 12;
  const qkv = 2 * n * nEmbd * (3 * nEmbd) * nLayer;
  const outp = 2 * n * nEmbd * nEmbd * nLayer;
  const ffn = 2 * n * nEmbd * nFf * nLayer;
  return {
    qkv: stageGflops(snap, "qkv", qkv, peak),
    out_proj: stageGflops(snap, "out_proj", outp, peak),
    ffn_gate: stageGflops(snap, "ffn_gate", ffn, peak),
    ffn_up: stageGflops(snap, "ffn_up", ffn, peak),
    ffn_down: stageGflops(snap, "ffn_down", ffn, peak),
    attn_qk: stageGflops(snap, "attn_qk", qkFlops, peak),
    attn_vmix: stageGflops(snap, "attn_vmix", vmixFlops, peak),
    matmul_all: {
      ms: MATMUL_STAGES.reduce((s, k) => s + (snap.stages_ms?.[k] ?? 0), 0),
      flops: matmulFlops,
      gflops: gflops(
        matmulFlops,
        MATMUL_STAGES.reduce((s, k) => s + (snap.stages_ms?.[k] ?? 0), 0),
      ),
    },
    peak_gflops: peak,
  };
}

function pickCases() {
  const singleN = BASELINE.single?.n ?? 8;
  const single = CORPUS.cases.slice(0, singleN);
  const must = new Set(["short-hello-document", "short-hello-none", "long-repeated"]);
  const extra = CORPUS.cases.filter((c) => must.has(c.id) && !single.some((s) => s.id === c.id));
  const profileSet = [...single, ...extra];
  const seen = new Set();
  const unique = [];
  for (const c of [...profileSet, ...CORPUS.cases]) {
    if (seen.has(c.id)) continue;
    seen.add(c.id);
    unique.push(c);
  }
  return { single, batched: CORPUS.cases, profile: unique };
}

async function profileCase(runner, c, repeats) {
  const runs = [];
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    const got = await runner.profile(c.text, c.prefix);
    const wall = performance.now() - t0;
    runs.push({ got, wall });
  }
  runs.sort((a, b) => a.got.profile.total_ms - b.got.profile.total_ms);
  const mid = runs[Math.floor(runs.length / 2)];
  return {
    id: c.id,
    prefix: c.prefix,
    n_tokens: mid.got.profile.n_tokens,
    n_ids_golden: nIds(c.id),
    repeats,
    js_wall_ms: mid.got.js_wall_ms ?? mid.wall,
    rust_total_ms: mid.got.profile.total_ms,
    js_minus_rust_ms: (mid.got.js_wall_ms ?? mid.wall) - mid.got.profile.total_ms,
    snapshot: mid.got.profile,
    vector: mid.got.vector,
    summary: summarizeStages(mid.got.profile),
  };
}

function parseLlamaPerf(stderr) {
  const text = stderr || "";
  const grab = (re) => {
    const m = text.match(re);
    return m ? Number(m[1]) : null;
  };
  return {
    load_ms: grab(/load time\s*=\s*([\d.]+)\s*ms/),
    prompt_eval_ms: grab(/prompt eval time\s*=\s*([\d.]+)\s*ms/),
    eval_ms: grab(/eval time\s*=\s*([\d.]+)\s*ms/),
    total_ms: grab(/total time\s*=\s*([\d.]+)\s*ms/),
  };
}

function runLlamaTimedSync(prefixed) {
  const paths = resolvePaths();
  if (!existsSync(paths.bin)) return { ok: false, error: `llama-embedding missing at ${paths.bin}` };
  const promptDir = join(tmpdir(), "milton-harness");
  mkdirSync(promptDir, { recursive: true });
  const promptFile = join(
    promptDir,
    `prompt-profile-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}.txt`,
  );
  writeFileSync(promptFile, prefixed);
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
  ];
  const binDir = dirname(paths.bin);
  const t0 = performance.now();
  const r = spawnSync(paths.bin, args, {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    env: {
      ...process.env,
      LD_LIBRARY_PATH: `${binDir}${process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : ""}`,
    },
  });
  const wall = performance.now() - t0;
  try {
    unlinkSync(promptFile);
  } catch {
    /* tmp */
  }
  return {
    ok: r.status === 0,
    wall_ms: wall,
    perf: parseLlamaPerf(r.stderr || ""),
    status: r.status,
    stderr_tail: (r.stderr || "").slice(-800),
  };
}

function simdDump(wasmPath) {
  try {
    const raw = execFileSync("python3", [join(HERE, "wasm-simd-per-fn.py"), wasmPath], {
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    });
    return JSON.parse(raw);
  } catch (e) {
    return { error: e.message };
  }
}

function mdTable(rows, cols) {
  const head = `| ${cols.map((c) => c.header).join(" | ")} |`;
  const sep = `| ${cols.map(() => "---").join(" | ")} |`;
  const body = rows.map((r) => `| ${cols.map((c) => c.cell(r)).join(" | ")} |`).join("\n");
  return `${head}\n${sep}\n${body}`;
}

function fmt(n, d = 3) {
  if (n == null || !Number.isFinite(n)) return "—";
  return Number(n).toFixed(d);
}

function fmtPct(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  return `${n.toFixed(1)}%`;
}

async function main() {
  if (!existsSync(GGUF)) {
    console.error(`fail-closed: GGUF not found at ${GGUF}`);
    process.exit(2);
  }

  const host = hostInfo();
  const cases = pickCases();
  console.error("building native profile + default milton-embed + profile wasm...");
  const nativeProfileBin = buildNativeProfile();
  const defaultEmbedBin = buildDefaultEmbed();
  const profileWasmDir = buildProfileWasm();
  const committedWasm = join(ROOT, "wasm", "milton_bg.wasm");
  const committedBytes = existsSync(committedWasm) ? statSync(committedWasm).size : null;
  const committedSha = existsSync(committedWasm)
    ? execFileSync("sha256sum", [committedWasm], { encoding: "utf8" }).split(" ")[0]
    : null;

  const nativeProf = createNativeProfiler(nativeProfileBin);
  const wasmProf = await createWasmProfiler(profileWasmDir);
  const nativeDefault = createNativeEmbedder();

  const nativeRows = [];
  const wasmRows = [];
  const matchChecks = [];

  try {
    // Warmup
    await nativeProf.profile("hello", "none");
    wasmProf.profile("hello", "none");
    await nativeDefault.embed("hello", { prefix: "none" });

    for (const c of cases.profile) {
      const repeats = c.id === "long-repeated" ? 1 : 3;
      console.error(`profile ${c.id} n=${nIds(c.id)} repeats=${repeats}`);
      const nat = await profileCase(nativeProf, c, repeats);
      const was = await profileCase(
        { profile: (t, p) => Promise.resolve(wasmProf.profile(t, p)) },
        c,
        repeats,
      );
      nat.roofline = attachRoofline(nat.snapshot, host.roofline.avx2_peak_gflops);
      was.roofline = attachRoofline(was.snapshot, host.roofline.simd128_peak_gflops);
      const def = await nativeDefault.embed(c.text, { prefix: c.prefix });
      const vsNative = vecEqual(Float32Array.from(nat.vector), def);
      const vsWasm = vecEqual(Float32Array.from(was.vector), def);
      matchChecks.push({
        id: c.id,
        profiled_native_vs_default: vsNative,
        profiled_wasm_vs_default_native: vsWasm,
      });
      // Drop vectors from the receipt (large).
      delete nat.vector;
      delete was.vector;
      nativeRows.push(nat);
      wasmRows.push(was);
    }
  } finally {
    nativeProf.close();
    nativeDefault.close();
  }

  // Milton NATIVE (default, no timers) vs llama.cpp n_threads=1 on the wasm:bench sets.
  process.env.MILTON_EMBED_BIN = defaultEmbedBin;
  const nativeThru = createNativeEmbedder();
  let nativeSingle = null;
  let nativeBatched = null;
  try {
    await nativeThru.embed(cases.single[0].text, { prefix: cases.single[0].prefix });
    const t0 = performance.now();
    for (const c of cases.single) {
      await nativeThru.embed(c.text, { prefix: c.prefix });
    }
    const singleMs = performance.now() - t0;
    nativeSingle = {
      n: cases.single.length,
      ms: singleMs,
      embeddings_per_sec: (cases.single.length / singleMs) * 1000,
    };
    const t1 = performance.now();
    for (const c of cases.batched) {
      await nativeThru.embed(c.text, { prefix: c.prefix });
    }
    const batchedMs = performance.now() - t1;
    nativeBatched = {
      n: cases.batched.length,
      ms: batchedMs,
      embeddings_per_sec: (cases.batched.length / batchedMs) * 1000,
    };
  } finally {
    nativeThru.close();
  }

  const llamaCases = ["short-hello-document", "long-repeated"];
  const llamaRows = [];
  const paths = resolvePaths();
  let llamaSingle = null;
  if (existsSync(paths.bin)) {
    for (const id of llamaCases) {
      const c = CORPUS.cases.find((x) => x.id === id);
      const prefixed = applyPrefix(c.text, c.prefix);
      // Discard first (load). Second is compute-ish; also parse prompt eval.
      runLlamaTimedSync(prefixed);
      const second = runLlamaTimedSync(prefixed);
      llamaRows.push({
        id,
        n_tokens: nIds(id),
        ...second,
      });
    }
    // Throughput on the same 8-case single set, using parsed prompt-eval (excludes reload).
    let promptSum = 0;
    let nOk = 0;
    for (const c of cases.single) {
      const r = runLlamaTimedSync(applyPrefix(c.text, c.prefix));
      if (r.ok && r.perf.prompt_eval_ms != null) {
        promptSum += r.perf.prompt_eval_ms;
        nOk += 1;
      }
    }
    if (nOk === cases.single.length) {
      llamaSingle = {
        n: nOk,
        prompt_eval_ms: promptSum,
        embeddings_per_sec: (nOk / promptSum) * 1000,
        method: "sum of llama-embedding 'prompt eval time' with -t 1 (model already resident after first call; each spawn still reloads — use prompt-eval, not wall)",
      };
    }
  } else {
    llamaRows.push({ error: "llama-embedding not built; run npm run harness:setup" });
  }

  const wasmSimd = existsSync(committedWasm) ? simdDump(committedWasm) : { error: "no committed wasm" };
  const hotNames = [
    "gemv_q4_k_8x8_q8_k",
    "vec_dot_q5_k_q8_k",
    "vec_dot_q6_k_q8_k",
    "vmix_axpy",
    "expf",
    "softmax",
    "attention",
    "matmul_ggml",
    "quantize_row_q8",
    "rope",
    "layer_norm",
    "sincos",
  ];
  let simdHot = [];
  if (wasmSimd.functions) {
    simdHot = wasmSimd.functions
      .filter((f) => hotNames.some((h) => f.name.toLowerCase().includes(h.replaceAll("_", ""))) || hotNames.some((h) => f.name.includes(h)))
      .slice(0, 40);
    if (!simdHot.length) {
      simdHot = wasmSimd.functions.filter((f) => f.simd_insns > 0).slice(0, 25);
    }
  }

  const shortNat = nativeRows.find((r) => r.id === "short-hello-document");
  const longNat = nativeRows.find((r) => r.id === "long-repeated");
  const shortWasm = wasmRows.find((r) => r.id === "short-hello-document");
  const longWasm = wasmRows.find((r) => r.id === "long-repeated");

  const h1 = (() => {
    if (!shortNat || !longNat) return { status: "missing" };
    const sN = shortNat.n_tokens;
    const lN = longNat.n_tokens;
    const sMat = shortNat.summary.matmul_ms;
    const lMat = longNat.summary.matmul_ms;
    const msPerTokShort = sMat / sN;
    const msPerTokLong = lMat / lN;
    const bytesPerTok = longNat.snapshot.counters.gemv_weight_bytes / lN;
    const callsPerTok = longNat.snapshot.counters.gemv_calls / lN;
    return {
      predict: "time proportional to tokens x weights if each token re-reads every weight block",
      gemv_calls_per_token: callsPerTok,
      weight_bytes_per_token: bytesPerTok,
      native_matmul_ms_per_token_short: msPerTokShort,
      native_matmul_ms_per_token_long: msPerTokLong,
      native_matmul_ms_ratio_long_over_short: lMat / sMat,
      token_ratio: lN / sN,
      linear_in_n: Math.abs(lMat / sMat / (lN / sN) - 1) < 0.35,
      wasm_matmul_ms_per_token_short: shortWasm.summary.matmul_ms / shortWasm.n_tokens,
      wasm_matmul_ms_per_token_long: longWasm.summary.matmul_ms / longWasm.n_tokens,
    };
  })();

  const h2 = (() => {
    if (!shortNat || !longNat) return { status: "missing" };
    const sN = shortNat.n_tokens;
    const lN = longNat.n_tokens;
    const n2 = (lN * lN) / (sN * sN);
    return {
      predict: "attention / V-mix O(n²) scalar or re-allocate per token dominates",
      native_attn_pct_short: shortNat.summary.attn_pct,
      native_attn_pct_long: longNat.summary.attn_pct,
      native_attn_ms_ratio: longNat.summary.attn_ms / shortNat.summary.attn_ms,
      n_squared_ratio: n2,
      wasm_attn_pct_short: shortWasm.summary.attn_pct,
      wasm_attn_pct_long: longWasm.summary.attn_pct,
      scores_alloc: "one Vec<n_tokens> per attention call, not per token (see ops.rs attention_named)",
    };
  })();

  const h3 = (() => {
    const jsOverheadShort = shortWasm ? shortWasm.js_minus_rust_ms : null;
    const jsOverheadLong = longWasm ? longWasm.js_minus_rust_ms : null;
    const rustVsJsLongPct =
      longWasm && longWasm.js_wall_ms > 0
        ? (100 * longWasm.rust_total_ms) / longWasm.js_wall_ms
        : null;
    return {
      predict: "JS boundary copies / allocation per call dominate the WASM path",
      wasm_js_minus_rust_ms_short: jsOverheadShort,
      wasm_js_minus_rust_ms_long: jsOverheadLong,
      rust_share_of_js_wall_long_pct: rustVsJsLongPct,
      one_call_per_embed: true,
      embedding_bytes: 768 * 4,
    };
  })();

  const h4 = {
    predict: "a hot path with no SIMD128 (opcode count is not coverage)",
    committed_wasm_functions: wasmSimd.n_functions ?? null,
    functions_with_simd: wasmSimd.n_functions_with_simd ?? null,
    simd_insn_total: wasmSimd.simd_insn_total ?? null,
    hot_or_simd_functions: simdHot,
  };

  const nativeVsLlama = {
    milton_native_single: nativeSingle,
    milton_native_batched: nativeBatched,
    llama_single: llamaSingle,
    llama_case_rows: llamaRows,
    ratio_llama_over_milton_native_single:
      llamaSingle && nativeSingle
        ? llamaSingle.embeddings_per_sec / nativeSingle.embeddings_per_sec
        : null,
    stored_baseline_llama_single_emb_s: BASELINE.single_thread?.single?.embeddings_per_sec ?? null,
  };

  const receipt = {
    schema: "milton.stage-profile/1",
    issue: 28,
    refs: [25, 26, 24],
    host,
    gguf: GGUF,
    wasm_footprint: {
      committed_bytes: committedBytes,
      committed_sha256: committedSha,
      profile_wasm_bytes: wasmProf.wasmBytes,
      note: "profile wasm is feature-gated and is NOT the shipped artifact. Default wasm:build must stay bit-identical to 49a6e8d2.",
    },
    cases: {
      native: nativeRows,
      wasm: wasmRows,
    },
    vector_match: matchChecks,
    hypotheses: { H1: h1, H2: h2, H3: h3, H4: h4 },
    native_vs_llamacpp: nativeVsLlama,
    simd: {
      n_functions: wasmSimd.n_functions,
      n_functions_with_simd: wasmSimd.n_functions_with_simd,
      simd_insn_total: wasmSimd.simd_insn_total,
      top_simd: (wasmSimd.functions || []).filter((f) => f.simd_insns > 0).slice(0, 30),
      no_simd_large: (wasmSimd.functions || [])
        .filter((f) => f.simd_insns === 0 && f.insns >= 80)
        .slice(0, 30),
    },
  };

  mkdirSync(dirname(OUT_JSON), { recursive: true });
  writeFileSync(OUT_JSON, `${JSON.stringify(receipt, null, 2)}\n`);

  const tableCols = [
    { header: "case", cell: (r) => r.id },
    { header: "n", cell: (r) => String(r.n_tokens) },
    { header: "total ms", cell: (r) => fmt(r.rust_total_ms, 2) },
    { header: "matmul %", cell: (r) => fmtPct(r.summary.matmul_pct) },
    { header: "attn %", cell: (r) => fmtPct(r.summary.attn_pct) },
    { header: "qkv ms", cell: (r) => fmt(r.snapshot.stages_ms.qkv, 2) },
    { header: "ffn* ms", cell: (r) => fmt((r.snapshot.stages_ms.ffn_gate || 0) + (r.snapshot.stages_ms.ffn_up || 0) + (r.snapshot.stages_ms.ffn_down || 0), 2) },
    { header: "qk ms", cell: (r) => fmt(r.snapshot.stages_ms.attn_qk, 2) },
    { header: "softmax ms", cell: (r) => fmt(r.snapshot.stages_ms.attn_softmax, 2) },
    { header: "V-mix ms", cell: (r) => fmt(r.snapshot.stages_ms.attn_vmix, 2) },
    { header: "RoPE ms", cell: (r) => fmt(r.snapshot.stages_ms.rope, 3) },
    { header: "LN ms", cell: (r) => fmt(r.snapshot.stages_ms.layernorm, 3) },
    { header: "tok+emb+pool", cell: (r) => fmt((r.snapshot.stages_ms.tokenize || 0) + (r.snapshot.stages_ms.embedding_lookup || 0) + (r.snapshot.stages_ms.pooling || 0), 3) },
  ];

  const md = [];
  md.push("## Native (AVX2, `--features profile`)");
  md.push(mdTable(nativeRows, tableCols));
  md.push("");
  md.push("## WASM (SIMD128, `--features profile`, separate artifact)");
  md.push(mdTable(wasmRows, tableCols));
  md.push("");
  writeFileSync(join(HERE, "..", "goldens", "stage-profile.tables.md"), `${md.join("\n")}\n`);
  process.stdout.write(md.join("\n") + "\n");
  process.stdout.write(`\nwrote ${OUT_JSON}\n`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
