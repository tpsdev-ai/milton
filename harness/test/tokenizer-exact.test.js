import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus, corpusDigest } from "../lib/corpus.js";
import { PREFIX } from "../lib/prefix.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const PIN_PATH = join(ROOT, "harness", "goldens", "tokenizer-pin.json");
const TOKENS_PATH = join(ROOT, "harness", "goldens", "tokens.json");

function sha256file(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

describe("tokenizer goldens (pinned HF nomic WordPiece)", () => {
  const pin = JSON.parse(readFileSync(PIN_PATH, "utf8"));
  const gold = JSON.parse(readFileSync(TOKENS_PATH, "utf8"));
  const corpus = loadCorpus();

  it("records tokenizer source + digest (an unpinned reference is not a reference)", () => {
    assert.equal(pin.schema, "milton.tokenizer-pin/1");
    assert.equal(pin.source.repo, "nomic-ai/nomic-embed-text-v1.5");
    assert.match(pin.source.revision, /^[0-9a-f]{40}$/);
    assert.match(pin.source.files["tokenizer.json"].sha256, /^[0-9a-f]{64}$/);
    assert.match(pin.source.files["vocab.txt"].sha256, /^[0-9a-f]{64}$/);
    assert.equal(pin.n_cases, corpus.cases.length);
  });

  it("pin digests match the committed tokenizer files", () => {
    for (const [name, meta] of Object.entries(pin.source.files)) {
      const path = join(ROOT, meta.path);
      assert.ok(existsSync(path), `missing ${name} at ${meta.path}`);
      assert.equal(sha256file(path), meta.sha256, `${name} digest drifted from pin`);
    }
  });

  it("token goldens cover every corpus case and share the corpus digest", () => {
    assert.equal(gold.schema, "milton.token-goldens/1");
    assert.equal(gold.n, corpus.cases.length);
    assert.equal(gold.corpus_digest, corpusDigest(corpus));
    assert.equal(gold.corpus_digest, pin.corpus_digest);
    const ids = new Set(gold.items.map((it) => it.id));
    for (const c of corpus.cases) {
      assert.ok(ids.has(c.id), `golden missing ${c.id}`);
    }
  });

  it("prefix bytes on goldens match harness PREFIX (space after colon)", () => {
    assert.equal(PREFIX.document, "search_document: ");
    assert.equal(PREFIX.query, "search_query: ");
    for (const item of gold.items) {
      const p = PREFIX[item.prefix];
      assert.ok(item.prefixed.startsWith(p), `${item.id} prefixed bytes drifted`);
    }
  });

  it("empty-document goldens are the prefix tokens, not a guessed zero sequence", () => {
    const emptyDoc = gold.items.find((it) => it.id === "empty-document");
    const emptyNone = gold.items.find((it) => it.id === "empty-none");
    assert.deepEqual(emptyNone.ids, [101, 102]);
    assert.ok(emptyDoc.ids.length > emptyNone.ids.length);
    assert.deepEqual(emptyDoc.tokens.slice(0, 5), ["[CLS]", "search", "_", "document", ":"]);
  });
});
