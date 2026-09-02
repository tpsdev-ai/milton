#!/usr/bin/env node
/**
 * Install footprint: shipped package size and "zero native binary in src/".
 */
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const SRC = join(ROOT, "src");
const WASM = join(ROOT, "wasm");

const NATIVE_RE = /\.(node|so|dylib|dll|a)$/i;
const NATIVE_NAMES = /^(llama|onnxruntime|ggml)/i;

function walk(dir, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else acc.push({ path: p, size: st.size, name });
  }
  return acc;
}

const srcFiles = walk(SRC);
const wasmFiles = walk(WASM);
const nativeInSrc = srcFiles.filter(
  (f) => NATIVE_RE.test(f.name) || NATIVE_NAMES.test(f.name),
);
const nativeInWasm = wasmFiles.filter(
  (f) => NATIVE_RE.test(f.name) || NATIVE_NAMES.test(f.name),
);

let packedBytes = null;
let packedMb = null;
try {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: ROOT,
    encoding: "utf8",
  });
  const parsed = JSON.parse(out);
  const entry = Array.isArray(parsed) ? parsed[0] : parsed;
  packedBytes = entry.size ?? entry.unpackedSize ?? null;
  packedMb = packedBytes != null ? packedBytes / (1024 * 1024) : null;
} catch (err) {
  process.stderr.write(`npm pack --dry-run failed: ${err.message}\n`);
}

const srcBytes = srcFiles.reduce((s, f) => s + f.size, 0);
const wasmBytes = wasmFiles.reduce((s, f) => s + f.size, 0);

const report = {
  schema: "milton.footprint/1",
  host: {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
  },
  src: {
    files: srcFiles.map((f) => ({ path: relative(ROOT, f.path), bytes: f.size })),
    bytes: srcBytes,
    mb: srcBytes / (1024 * 1024),
    native_binaries: nativeInSrc.map((f) => relative(ROOT, f.path)),
    native_binary_count: nativeInSrc.length,
  },
  wasm: {
    files: wasmFiles.map((f) => ({ path: relative(ROOT, f.path), bytes: f.size })),
    bytes: wasmBytes,
    mb: wasmBytes / (1024 * 1024),
    native_binaries: nativeInWasm.map((f) => relative(ROOT, f.path)),
    native_binary_count: nativeInWasm.length,
  },
  shipped_mb: (srcBytes + wasmBytes) / (1024 * 1024),
  npm_pack_dry_run: {
    bytes: packedBytes,
    mb: packedMb,
  },
  // issue #23 addendum: SIMD code may grow the .wasm modestly; report delta vs #22, keep well under 1 MB.
  vs_issue22: {
    wasm_milton_bg_bytes_was: 629953,
    wasm_milton_bg_bytes_now: wasmFiles.find((f) => f.name === "milton_bg.wasm")?.size ?? null,
    wasm_delta_bytes:
      (wasmFiles.find((f) => f.name === "milton_bg.wasm")?.size ?? 0) - 629953,
    pack_mb_now: packedMb,
    well_under_1mb: packedMb != null && packedMb < 1,
  },
  assert: {
    zero_native_in_src: nativeInSrc.length === 0,
    zero_native_in_wasm: nativeInWasm.length === 0,
  },
};

mkdirSync(join(HERE, "..", "receipts"), { recursive: true });
writeFileSync(join(HERE, "..", "receipts", "footprint.json"), `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(JSON.stringify(report, null, 2) + "\n");

if (nativeInSrc.length !== 0 || nativeInWasm.length !== 0) {
  process.stderr.write("FAIL: native binary found under src/ or wasm/\n");
  process.exitCode = 1;
}
