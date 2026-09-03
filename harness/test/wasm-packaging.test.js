import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const WASM_DIR = join(ROOT, "wasm");
const WASM = join(WASM_DIR, "milton_bg.wasm");
const WASM_T = join(WASM_DIR, "milton_threads_bg.wasm");
const GLUE = join(WASM_DIR, "milton.js");
const GLUE_T = join(WASM_DIR, "milton_threads.js");
const SRC = join(ROOT, "src", "index.js");
const CONFIG = join(ROOT, "crate", ".cargo", "config.toml");
const BUILD = join(ROOT, "scripts", "build-wasm.sh");

function wasmImportsSharedMemory(bytes) {
  if (bytes.subarray(0, 4).toString() !== "\0asm") return false;
  let i = 8;
  const uvarint = () => {
    let x = 0;
    let s = 0;
    while (i < bytes.length) {
      const c = bytes[i];
      i += 1;
      x |= (c & 0x7f) << s;
      if (c < 0x80) return x;
      s += 7;
    }
    return 0;
  };
  while (i < bytes.length) {
    const sid = uvarint();
    const slen = uvarint();
    const start = i;
    const end = i + slen;
    if (sid === 2) {
      let j = start;
      const uv = () => {
        let x = 0;
        let s = 0;
        while (j < end) {
          const c = bytes[j];
          j += 1;
          x |= (c & 0x7f) << s;
          if (c < 0x80) return x;
          s += 7;
        }
        return 0;
      };
      const n = uv();
      for (let k = 0; k < n; k += 1) {
        const ml = uv();
        j += ml;
        const nl = uv();
        j += nl;
        const kind = bytes[j];
        j += 1;
        if (kind === 2) {
          const flags = bytes[j];
          return (flags & 2) !== 0;
        }
        break;
      }
    }
    i = end;
  }
  return false;
}

describe("WASM-SIMD packaging", () => {
  it("ships a prebuilt .wasm (no compile at consumer install)", () => {
    assert.ok(existsSync(WASM), `missing ${WASM} — builder must run npm run wasm:build`);
    assert.ok(existsSync(GLUE), `missing ${GLUE}`);
    const bytes = readFileSync(WASM);
    assert.ok(bytes.length > 1024, `wasm too small: ${bytes.length}`);
    assert.equal(bytes.subarray(0, 4).toString(), "\0asm");
  });

  it("ships a second prebuilt shared-memory .wasm for Node threads", () => {
    assert.ok(existsSync(WASM_T), `missing ${WASM_T} — builder must run npm run wasm:build`);
    assert.ok(existsSync(GLUE_T), `missing ${GLUE_T}`);
    const bytes = readFileSync(WASM_T);
    assert.ok(bytes.length > 1024, `threads wasm too small: ${bytes.length}`);
    assert.equal(bytes.subarray(0, 4).toString(), "\0asm");
    assert.equal(wasmImportsSharedMemory(readFileSync(WASM)), false);
    assert.equal(wasmImportsSharedMemory(bytes), true);
  });

  it("was compiled with WASM SIMD (0xfd opcodes present)", () => {
    const bytes = readFileSync(WASM);
    let n = 0;
    for (const b of bytes) if (b === 0xfd) n += 1;
    assert.ok(n > 0, "milton_bg.wasm has no 0xfd SIMD opcodes");
  });

  it("builder rustflags request +simd128", () => {
    const cfg = readFileSync(CONFIG, "utf8");
    assert.match(cfg, /target-feature=\+simd128/);
    const build = readFileSync(BUILD, "utf8");
    assert.match(build, /target-feature=\+simd128/);
    assert.match(build, /atomics,\+bulk-memory/);
    assert.match(build, /Consumers do NOT run this/);
  });

  it("remap-path-prefix makes registry panic paths host-stable", () => {
    const build = readFileSync(BUILD, "utf8");
    assert.match(build, /CARGO_HOME_DIR="\$\{CARGO_HOME:-\$HOME\/\.cargo\}"/);
    assert.match(
      build,
      /--remap-path-prefix=\$\{CARGO_HOME_DIR\}\/registry\/src=\/cargo\/registry\/src/,
    );
    assert.match(build, /--remap-path-prefix=\$\{ROOT\}=\/milton/);
    assert.match(build, /--remap-path-prefix=\$\{RUSTUP_HOME_DIR\}=\/rustup/);
    const bytes = readFileSync(WASM);
    assert.ok(
      bytes.includes(Buffer.from("/cargo/registry/src")),
      "committed wasm missing remapped /cargo/registry/src",
    );
    assert.ok(
      bytes.includes(Buffer.from("/rustup")),
      "committed wasm missing remapped /rustup sysroot",
    );
    assert.equal(bytes.includes(Buffer.from("/usr/local/cargo")), false);
    assert.equal(bytes.includes(Buffer.from("/home/runner/.cargo")), false);
    assert.equal(bytes.includes(Buffer.from("/usr/local/rustup")), false);
    assert.equal(bytes.includes(Buffer.from("/home/runner/.rustup")), false);
  });

  it("public glue loads the prebuilt wasm, not a native bin", () => {
    const src = readFileSync(SRC, "utf8");
    assert.match(src, /milton_bg\.wasm/);
    assert.match(src, /milton_threads_bg\.wasm/);
    assert.match(src, /embed\(text, prefix\)/);
    assert.doesNotMatch(src, /milton-embed/);
    assert.doesNotMatch(src, /node-gyp/);
    assert.doesNotMatch(src, /spawn\(/);
    assert.doesNotMatch(src, /\.node['"]/);
  });

  it("wasm/ has no per-platform native artifacts", () => {
    const names = existsSync(WASM_DIR) ? readdirSync(WASM_DIR) : [];
    for (const name of names) {
      assert.doesNotMatch(name, /\.(node|so|dylib|dll)$/i);
    }
  });
});
