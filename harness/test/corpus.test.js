import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadCorpus, corpusDigest } from "../lib/corpus.js";

describe("conformance corpus", () => {
  const corpus = loadCorpus();

  it("covers the required trap classes", () => {
    const ids = new Set(corpus.cases.map((c) => c.id));
    for (const need of [
      "short-hello-document",
      "short-hello-query",
      "short-hello-none",
      "unicode-cjk-emoji",
      "unicode-nfc",
      "unicode-nfd",
      "whitespace-padded",
      "whitespace-only",
      "empty-document",
      "empty-none",
      "long-repeated",
      "flair-memory-preference",
      "flair-memory-decision",
      "flair-query-recall",
    ]) {
      assert.ok(ids.has(need), `missing ${need}`);
    }
  });

  it("documents a trap on every case and includes all three prefix kinds", () => {
    const prefixes = new Set(corpus.cases.map((c) => c.prefix));
    assert.deepEqual([...prefixes].sort(), ["document", "none", "query"]);
    for (const c of corpus.cases) {
      assert.ok(c.trap.length > 20, `${c.id} trap too thin`);
    }
  });

  it("has a stable content-address over {id,text,prefix}", () => {
    const d = corpusDigest(corpus);
    assert.match(d, /^[0-9a-f]{64}$/);
    assert.equal(corpusDigest(corpus), d);
  });

  it("keeps NFC and NFD as distinct committed bytes", () => {
    const nfc = corpus.cases.find((c) => c.id === "unicode-nfc");
    const nfd = corpus.cases.find((c) => c.id === "unicode-nfd");
    assert.notEqual(nfc.text, nfd.text);
    assert.equal(nfc.text.normalize("NFC"), nfd.text.normalize("NFC"));
  });
});
