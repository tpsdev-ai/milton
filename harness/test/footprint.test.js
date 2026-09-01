import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, statSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const SRC = join(ROOT, "src");
const WASM = join(ROOT, "wasm");
const PKG = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));

const NATIVE_RE = /\.(node|so|dylib|dll|a)$/i;

function walk(dir, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else acc.push(p);
  }
  return acc;
}

describe("light by construction", () => {
  it("ships src/ + prebuilt wasm/ — harness is not in package files", () => {
    assert.deepEqual(PKG.files, ["src", "wasm"]);
  });

  it("has no install / postinstall compile step", () => {
    assert.equal(PKG.scripts?.install, undefined);
    assert.equal(PKG.scripts?.postinstall, undefined);
    assert.equal(PKG.scripts?.preinstall, undefined);
    assert.doesNotMatch(JSON.stringify(PKG.scripts ?? {}), /node-gyp|cargo build|wasm-pack/i);
  });

  it("src/ contains zero native binaries", () => {
    const native = walk(SRC).filter((p) => NATIVE_RE.test(p));
    assert.deepEqual(native, []);
  });

  it("wasm/ contains zero native binaries (prebuilt .wasm only)", () => {
    const native = walk(WASM).filter((p) => NATIVE_RE.test(p));
    assert.deepEqual(native, []);
  });

  it("src/ is glue only — no heavy native runtime, no compiled binaries", () => {
    const src = readFileSync(join(SRC, "index.js"), "utf8");
    assert.match(src, /embed\(text, prefix\)/);
    assert.doesNotMatch(src, /onnxruntime|llama\.cpp/i);
    assert.match(src, /milton_bg\.wasm/);
  });
});
