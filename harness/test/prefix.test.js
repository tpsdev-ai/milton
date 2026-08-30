import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { applyPrefix, PREFIX } from "../lib/prefix.js";

describe("Flair nomic prefix convention", () => {
  it("uses a space after the colon (HFE NOMIC_TEMPLATES)", () => {
    assert.equal(PREFIX.document, "search_document: ");
    assert.equal(PREFIX.query, "search_query: ");
    assert.equal(applyPrefix("hello", "document"), "search_document: hello");
    assert.equal(applyPrefix("hello", "query"), "search_query: hello");
    assert.equal(applyPrefix("hello", "none"), "hello");
  });

  it("concatenates unconditionally — already-prefixed text is double-prefixed", () => {
    assert.equal(
      applyPrefix("search_document: already prefixed", "document"),
      "search_document: search_document: already prefixed",
    );
  });

  it("preserves empty and whitespace bodies", () => {
    assert.equal(applyPrefix("", "document"), "search_document: ");
    assert.equal(applyPrefix("  x  ", "query"), "search_query:   x  ");
    assert.equal(applyPrefix("", "none"), "");
  });

  it("rejects the prefix STRING used as the kind (the silent inversion bug)", () => {
    assert.throws(() => applyPrefix("hello", "search_document"), /invalid kind/);
    assert.throws(() => applyPrefix("hello", "search_query"), /invalid kind/);
  });
});
