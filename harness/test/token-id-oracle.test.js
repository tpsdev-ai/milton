import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compareVectors } from "../lib/metrics.js";
import { loadEpsilon, loadGoldens } from "../lib/goldens.js";
import { TOKEN_ID_ORACLE_CASES, TOKEN_ID_ORACLE_SOURCE, TOKEN_ID_ORACLE_TOOL } from "../lib/token-id-oracle.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDENS = join(HERE, "..", "goldens");

describe("token-id oracle (#15): HF IDs through llama.cpp, not Milton", () => {
  const goldens = loadGoldens();
  const f16 = JSON.parse(readFileSync(join(GOLDENS, "vectors-f16.json"), "utf8"));
  const tokens = JSON.parse(readFileSync(join(GOLDENS, "tokens.json"), "utf8"));
  const old = JSON.parse(readFileSync(join(GOLDENS, "vectors-pre-15-wrong.json"), "utf8"));
  const eps = loadEpsilon();
  const budget = JSON.parse(readFileSync(join(GOLDENS, "quant-budget.json"), "utf8"));
  const byId = new Map(goldens.items.map((it) => [it.id, it]));
  const f16ById = new Map(f16.items.map((it) => [it.id, it]));
  const tokById = new Map(tokens.items.map((it) => [it.id, it]));
  const oldQ4 = new Map(old.q4.map((it) => [it.id, it]));
  const oldF16 = new Map(old.f16.map((it) => [it.id, it]));

  it("does not loosen epsilon.json", () => {
    assert.equal(eps.epsilon, 1e-6);
    assert.equal(eps.epsilon_abs, 1e-5);
  });

  it("gates all 18 cases; the 2 #15 cases are no longer excluded", () => {
    assert.deepEqual(TOKEN_ID_ORACLE_CASES, ["unicode-nfd", "newlines-tabs"]);
    assert.deepEqual(budget.pending_excluded, []);
    assert.equal(budget.n, 18);
    assert.equal(budget.n_gated, 18);
    assert.equal(budget.ratio_max, 1.5);
    for (const id of TOKEN_ID_ORACLE_CASES) {
      const row = budget.per_case.find((p) => p.id === id);
      assert.ok(row, id);
      assert.equal(row.pending_issue, null, id);
    }
  });

  it("records independent provenance (embed-from-token-ids + tokens.json, not Milton)", () => {
    assert.equal(goldens.token_id_oracle?.tool, TOKEN_ID_ORACLE_TOOL);
    assert.equal(goldens.token_id_oracle?.token_ids_source, TOKEN_ID_ORACLE_SOURCE);
    assert.deepEqual(goldens.token_id_oracle?.cases, TOKEN_ID_ORACLE_CASES);
    assert.match(goldens.token_id_oracle?.note ?? "", /Not the llama-embedding text path/);
    for (const id of TOKEN_ID_ORACLE_CASES) {
      const it = byId.get(id);
      assert.ok(it?.provenance, id);
      assert.equal(it.provenance.oracle, "embed-from-token-ids", id);
      assert.equal(it.provenance.tool, TOKEN_ID_ORACLE_TOOL, id);
      assert.equal(it.provenance.token_ids_source, TOKEN_ID_ORACLE_SOURCE, id);
      assert.deepEqual(it.provenance.ids, tokById.get(id).ids, id);
      assert.doesNotMatch(JSON.stringify(it.provenance), /milton|crate\/src/i);
    }
    const src = readFileSync(join(HERE, "..", "scripts", "generate-goldens.mjs"), "utf8");
    assert.match(src, /runEmbedFromTokenIds/);
    assert.match(src, /TOKEN_ID_ORACLE_CASES/);
    assert.doesNotMatch(src, /milton-embed|Model::load|crate\/target/);
  });

  it("OLD text-path goldens FAIL the Q4-vs-Q4 gate against the new independent oracle", () => {
    for (const id of TOKEN_ID_ORACLE_CASES) {
      const neu = byId.get(id);
      const stale = oldQ4.get(id);
      assert.ok(neu && stale, id);
      const cmp = compareVectors(stale.vector, neu.vector, {
        epsilon: eps.epsilon,
        epsilonAbs: eps.epsilon_abs,
      });
      assert.equal(cmp.pass, false, `${id} old vs new should FAIL (old goldens were wrong)`);
      assert.ok(cmp.cos_dist > eps.epsilon, `${id} cos_dist=${cmp.cos_dist}`);
    }
  });

  it("unicode-nfd now matches unicode-nfc (same HF token IDs after strip_accents)", () => {
    const nfcTok = tokById.get("unicode-nfc");
    const nfdTok = tokById.get("unicode-nfd");
    assert.deepEqual(nfdTok.ids, nfcTok.ids);
    const nfc = byId.get("unicode-nfc");
    const nfd = byId.get("unicode-nfd");
    const cmp = compareVectors(nfd.vector, nfc.vector, {
      epsilon: eps.epsilon,
      epsilonAbs: eps.epsilon_abs,
    });
    assert.equal(cmp.pass, true, `nfd vs nfc cos_dist=${cmp.cos_dist} max_abs=${cmp.max_abs}`);
    const oldNfd = oldQ4.get("unicode-nfd");
    const oldVsNfc = compareVectors(oldNfd.vector, nfc.vector, {
      epsilon: eps.epsilon,
      epsilonAbs: eps.epsilon_abs,
    });
    assert.equal(oldVsNfc.pass, false, "old NFD golden must NOT match NFC (proves the old [UNK] path)");
  });

  it("F16 goldens for the 2 cases were also regenerated (same independence)", () => {
    for (const id of TOKEN_ID_ORACLE_CASES) {
      const neu = f16ById.get(id);
      const stale = oldF16.get(id);
      assert.ok(neu && stale, id);
      assert.equal(neu.provenance?.oracle, "embed-from-token-ids", id);
      const cmp = compareVectors(stale.vector, neu.vector, {
        epsilon: eps.epsilon,
        epsilonAbs: eps.epsilon_abs,
      });
      assert.equal(cmp.pass, false, `${id} old F16 vs new F16 should FAIL`);
    }
  });
});
