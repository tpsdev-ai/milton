#!/usr/bin/env node
/**
 * Package llama.cpp dump-dequant output into the committed golden fixture.
 * Oracle is the pinned llama.cpp to_float, not Milton.
 */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPin } from "../lib/goldens.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const GGUF = join(ROOT, "harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
const DUMP_BIN = join(ROOT, "harness/vendor/bin/dump-dequant");
const DUMP_DIR = join(ROOT, "harness/vendor/dequant-dump");
const SLICE = 256;

function sha256(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

function f32Array(buf) {
  if (buf.byteLength % 4 !== 0) throw new Error(`f32 dump length ${buf.byteLength} not divisible by 4`);
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
}

function hex(buf) {
  return Buffer.from(buf).toString("hex");
}

if (!existsSync(GGUF)) {
  console.error("fail-closed: GGUF missing — run npm run harness:setup");
  process.exit(1);
}

execFileSync("bash", [join(HERE, "build-dump-dequant.sh")], { stdio: "inherit" });
mkdirSync(DUMP_DIR, { recursive: true });
execFileSync(DUMP_BIN, [GGUF, DUMP_DIR], { stdio: "inherit" });

const dumpMeta = JSON.parse(readFileSync(join(DUMP_DIR, "meta.json"), "utf8"));
const pin = loadPin();
const ggufSha = sha256(readFileSync(GGUF));
if (ggufSha !== pin.gguf_sha256) {
  console.error(`fail-closed: GGUF sha256 ${ggufSha} != pin ${pin.gguf_sha256}`);
  process.exit(1);
}

const tensors = [];
for (const t of dumpMeta.tensors) {
  const f32 = readFileSync(join(DUMP_DIR, "tensors", `${t.name}.f32`));
  const wire = readFileSync(join(DUMP_DIR, "tensors", `${t.name}.wire`));
  const vals = f32Array(f32);
  if (vals.length !== t.n_elements) {
    throw new Error(`${t.name}: dumped ${vals.length} != n_elements ${t.n_elements}`);
  }
  const small = t.n_elements <= 2048;
  const midOff = Math.floor(t.n_elements / 2);
  tensors.push({
    name: t.name,
    type: t.type,
    shape: t.shape,
    n_elements: t.n_elements,
    n_bytes: t.n_bytes,
    head: vals.slice(0, small ? t.n_elements : SLICE),
    tail: small ? [] : vals.slice(-SLICE),
    mid: small ? [] : vals.slice(midOff, midOff + SLICE),
    mid_offset: small ? null : midOff,
    sha256_f32_le: sha256(f32),
    ...(small ? { wire_hex: hex(wire) } : {}),
  });
}

const kernel_blocks = [];
const kerDir = join(DUMP_DIR, "kernels");
for (const name of readdirSync(kerDir).filter((n) => n.endsWith(".type")).sort()) {
  const id = name.slice(0, -".type".length);
  const type = readFileSync(join(kerDir, `${id}.type`), "utf8");
  const n = Number(readFileSync(join(kerDir, `${id}.n`), "utf8"));
  const wire = readFileSync(join(kerDir, `${id}.wire`));
  const vals = f32Array(readFileSync(join(kerDir, `${id}.f32`)));
  if (vals.length !== n) throw new Error(`kernel ${id}: ${vals.length} != ${n}`);
  kernel_blocks.push({
    id,
    type,
    n_elements: n,
    wire_hex: hex(wire),
    values: vals,
  });
}

const poolingType = dumpMeta.metadata["nomic-bert.pooling_type"];
const pooling = poolingType === 1 ? "mean" : `unknown(${poolingType})`;
const normalization = {};
for (const [k, v] of Object.entries(dumpMeta.metadata)) {
  if (/norm/i.test(k)) normalization[k] = v;
}

const goldens = {
  schema: "milton.dequant/1",
  generated_at_utc: new Date().toISOString(),
  gguf_file: pin.gguf_file,
  gguf_sha256: pin.gguf_sha256,
  llamacpp_commit: pin.llamacpp_commit,
  llamacpp_digest: pin.llamacpp_digest,
  oracle: "llama.cpp ggml_get_type_traits()->to_float (pinned commit)",
  quant_types_present: dumpMeta.quant_types_present,
  metadata: {
    "general.architecture": dumpMeta.metadata["general.architecture"],
    "general.name": dumpMeta.metadata["general.name"],
    "general.file_type": dumpMeta.metadata["general.file_type"],
    "nomic-bert.block_count": dumpMeta.metadata["nomic-bert.block_count"],
    "nomic-bert.embedding_length": dumpMeta.metadata["nomic-bert.embedding_length"],
    "nomic-bert.context_length": dumpMeta.metadata["nomic-bert.context_length"],
    "nomic-bert.feed_forward_length": dumpMeta.metadata["nomic-bert.feed_forward_length"],
    "nomic-bert.attention.head_count": dumpMeta.metadata["nomic-bert.attention.head_count"],
    "nomic-bert.pooling_type": poolingType,
    "nomic-bert.attention.layer_norm_epsilon": dumpMeta.metadata["nomic-bert.attention.layer_norm_epsilon"],
  },
  pooling,
  pooling_type: poolingType,
  pooling_key: "nomic-bert.pooling_type",
  normalization_as_read: normalization,
  embedding_normalize_in_gguf: Object.keys(dumpMeta.metadata).some((k) =>
    /embd.*normaliz|normaliz.*embd/i.test(k),
  ),
  tensors,
  kernel_blocks,
};

writeFileSync(join(ROOT, "harness/goldens/dequant.json"), `${JSON.stringify(goldens, null, 2)}\n`);

// Epsilon: same discipline as harness/goldens/epsilon.json.
const COS_MARGIN = 10;
const ABS_MARGIN = 10;
const COS_MIN = 1e-6;
const ABS_MIN = 1e-5;
const observedMaxAbs = dumpMeta.run_to_run_max_abs ?? 0;
const observedMaxCos = 0;
const epsilon = Math.max(observedMaxCos * COS_MARGIN, COS_MIN);
const epsilonAbs = Math.max(observedMaxAbs * ABS_MARGIN, ABS_MIN);

const eps = {
  schema: "milton.dequant.epsilon/1",
  epsilon,
  epsilon_abs: epsilonAbs,
  derived_from: {
    method:
      "llama.cpp ggml to_float twice on the same GGUF tensor bytes (selected tensors + synth Q8_0/F16); measure max abs-diff",
    n_tensors: tensors.length,
    n_kernel_blocks: kernel_blocks.length,
    observed_max_cos_dist: observedMaxCos,
    observed_max_abs: observedMaxAbs,
    cos_margin: COS_MARGIN,
    abs_margin: ABS_MARGIN,
    cos_numeric_floor: COS_MIN,
    abs_numeric_floor: ABS_MIN,
    formula: "epsilon = max(observed_max_cos_dist * 10, 1e-6); epsilon_abs = max(observed_max_abs * 10, 1e-5)",
  },
};
writeFileSync(join(ROOT, "harness/goldens/dequant-epsilon.json"), `${JSON.stringify(eps, null, 2)}\n`);

process.stdout.write(
  JSON.stringify(
    {
      tensors: tensors.map((t) => ({ name: t.name, type: t.type, n: t.n_elements })),
      kernel_blocks: kernel_blocks.map((k) => ({ id: k.id, type: k.type, n: k.n_elements })),
      quant_types_present: dumpMeta.quant_types_present,
      pooling,
      epsilon,
      epsilon_abs: epsilonAbs,
      observed_max_abs: observedMaxAbs,
      gguf_sha256: ggufSha,
    },
    null,
    2,
  ) + "\n",
);
