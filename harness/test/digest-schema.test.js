import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadGoldens, referenceDigest } from "../lib/goldens.js";
import { sha256json } from "../lib/digest.js";

const HERE = dirname(fileURLToPath(import.meta.url));

describe("golden digest schema matches the gate", () => {
  const goldens = loadGoldens();
  const oracle = referenceDigest(goldens);

  it("hashes {id, prefix, dims, vector} — the payload the gate receipt uses", () => {
    assert.match(oracle, /^[0-9a-f]{64}$/);
    assert.equal(oracle, "8a860ab491b4ab626fa4f4fc31c349947c20782b59834942bae8d8e183296fd0");
  });

  it("the old {id, vector}-only hash does not match the oracle (would fail on the pre-fix generator)", () => {
    const stale = sha256json(goldens.items.map((it) => ({ id: it.id, vector: it.vector })));
    assert.notEqual(stale, oracle);
    assert.equal(stale, "a9f0f89bd39718cd7ade33ffa3ef606d04df68c2ca287cad3d0d43eacd7dcad2");
  });

  it("generate-goldens.mjs calls referenceDigest, not the stale payload", () => {
    const src = readFileSync(join(HERE, "..", "scripts", "generate-goldens.mjs"), "utf8");
    assert.match(src, /referenceDigest\(goldens\)/);
    assert.doesNotMatch(src, /id:\s*it\.id,\s*vector:\s*it\.vector/);
  });
});
