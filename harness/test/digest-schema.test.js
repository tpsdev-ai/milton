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
    assert.equal(oracle, "06a21ff7933f21ed170a2cdeaae06d8b9db7a6a6b5ac75816d28c467f0d9d6c5");
  });

  it("the old {id, vector}-only hash does not match the oracle (would fail on the pre-fix generator)", () => {
    const stale = sha256json(goldens.items.map((it) => ({ id: it.id, vector: it.vector })));
    assert.notEqual(stale, oracle);
    assert.equal(stale, "29261420294893312d724c7fe6fd9fd160d7ff5edd1ffb46d6235a1585f57d55");
  });

  it("generate-goldens.mjs calls referenceDigest, not the stale payload", () => {
    const src = readFileSync(join(HERE, "..", "scripts", "generate-goldens.mjs"), "utf8");
    assert.match(src, /referenceDigest\(goldens\)/);
    assert.doesNotMatch(src, /id:\s*it\.id,\s*vector:\s*it\.vector/);
  });
});
