import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const CRATE = join(ROOT, "crate");

function cargo(args) {
  return execFileSync("cargo", args, {
    cwd: CRATE,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

describe("dequant must-fail control — gate rejects a deliberately-wrong dequant", () => {
  it("pinned goldens exist and name the GGUF digest + quant types", () => {
    const g = JSON.parse(readFileSync(join(ROOT, "harness/goldens/dequant.json"), "utf8"));
    assert.equal(g.schema, "milton.dequant/1");
    assert.equal(g.gguf_sha256, "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac");
    assert.equal(g.pooling, "mean");
    assert.equal(g.pooling_type, 1);
    assert.ok(g.quant_types_present.Q4_K > 0);
    assert.ok(g.quant_types_present.F32 > 0);
    assert.ok(g.kernel_blocks.some((k) => k.type === "Q8_0"));
    assert.ok(g.kernel_blocks.some((k) => k.type === "F16"));
    assert.ok(g.kernel_blocks.some((k) => k.type === "Q4_K"));
  });

  it("correct dequant vs llama.cpp goldens is GREEN", () => {
    const out = cargo(["run", "--quiet", "--bin", "dequant-gate"]);
    const receipt = JSON.parse(out);
    assert.equal(receipt.result, "pass", JSON.stringify(receipt.failures, null, 2));
    assert.equal(receipt.failed, 0);
    assert.ok(receipt.n > 0);
  });

  it("wrong block scale / wrong type turn the run RED and name the failure", () => {
    const out = cargo(["run", "--quiet", "--bin", "dequant-must-fail"]);
    const receipt = JSON.parse(out);
    assert.equal(receipt.result, "pass", JSON.stringify(receipt, null, 2));
    assert.ok(receipt.named.some((s) => /wrong-block-scale RED/.test(s)), JSON.stringify(receipt.named));
    assert.ok(receipt.named.some((s) => /wrong-type RED/.test(s)), JSON.stringify(receipt.named));
    assert.deepEqual(receipt.slipped, []);
  });
});

describe("light by construction — crate is not a native runtime in src/", () => {
  it("src/ still has no native binaries after this slice", () => {
    assert.equal(existsSync(join(ROOT, "src", "index.js")), true);
    const src = readFileSync(join(ROOT, "src", "index.js"), "utf8");
    assert.match(src, /not implemented yet/);
  });
});
