import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const WASM_DIR = join(ROOT, "wasm");
const WASM = join(WASM_DIR, "milton_bg.wasm");
const GLUE = join(WASM_DIR, "milton.js");
const SRC = join(ROOT, "src", "index.js");
const CONFIG = join(ROOT, "crate", ".cargo", "config.toml");
const BUILD = join(ROOT, "scripts", "build-wasm.sh");

describe("WASM-SIMD packaging", () => {
  it("ships a prebuilt .wasm (no compile at consumer install)", () => {
    assert.ok(existsSync(WASM), `missing ${WASM} — builder must run npm run wasm:build`);
    assert.ok(existsSync(GLUE), `missing ${GLUE}`);
    const bytes = readFileSync(WASM);
    assert.ok(bytes.length > 1024, `wasm too small: ${bytes.length}`);
    assert.equal(bytes.subarray(0, 4).toString(), "\0asm");
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
    assert.match(build, /Consumers do NOT run this/);
  });

  it("public glue loads the prebuilt wasm, not a native bin", () => {
    const src = readFileSync(SRC, "utf8");
    assert.match(src, /milton_bg\.wasm/);
    assert.match(src, /embed\(text, prefix\)/);
    assert.doesNotMatch(src, /milton-embed/);
    assert.doesNotMatch(src, /node-gyp|\.node\b/);
    assert.doesNotMatch(src, /spawn\(/);
  });

  it("wasm/ has no per-platform native artifacts", () => {
    const names = existsSync(WASM_DIR) ? readdirSync(WASM_DIR) : [];
    for (const name of names) {
      assert.doesNotMatch(name, /\.(node|so|dylib|dll)$/i);
    }
  });
});
