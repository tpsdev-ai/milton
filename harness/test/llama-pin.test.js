import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPin } from "../lib/goldens.js";
import {
  assertLlamaAtPin,
  checkoutLlamaPin,
  llamaHead,
  pinnedLlamaCommit,
} from "../lib/llama-pin.js";

const HERE = dirname(fileURLToPath(import.meta.url));

function git(dir, args, opts = {}) {
  return execFileSync("git", ["-C", dir, ...args], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...opts,
  }).trim();
}

function leftoverRepo() {
  const dir = mkdtempSync(join(tmpdir(), "milton-llama-pin-"));
  execFileSync("git", ["init", dir], { stdio: "ignore" });
  git(dir, ["config", "user.email", "harness@example.test"]);
  git(dir, ["config", "user.name", "milton-harness"]);
  writeFileSync(join(dir, "README"), "commit-a\n");
  git(dir, ["add", "README"]);
  git(dir, ["commit", "-m", "A"]);
  const a = git(dir, ["rev-parse", "HEAD"]);
  writeFileSync(join(dir, "README"), "commit-b\n");
  git(dir, ["add", "README"]);
  git(dir, ["commit", "-m", "B"]);
  const b = git(dir, ["rev-parse", "HEAD"]);
  git(dir, ["checkout", "--detach", a]);
  return { dir, leftover: a, pin: b };
}

describe("llama.cpp pin — leftover tree must not become the oracle", () => {
  it("pin.json carries a full 40-char commit", () => {
    const pin = loadPin();
    assert.equal(pinnedLlamaCommit(pin), "61bdfd5298a78593be649a1035ee2a120b13c4f0");
    assert.throws(() => pinnedLlamaCommit({ llamacpp_commit: "b6399" }), /fail-closed/);
  });

  it("fail-closed when leftover HEAD != pin.json", () => {
    const { dir, leftover, pin } = leftoverRepo();
    try {
      assert.equal(llamaHead(dir), leftover);
      assert.throws(
        () => assertLlamaAtPin(dir, { llamacpp_commit: pin }),
        /leftover llama\.cpp tree/,
      );
      assert.equal(llamaHead(dir), leftover, "assert must not checkout");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("checkout moves a leftover tree onto the pin, then assert passes", () => {
    const { dir, leftover, pin } = leftoverRepo();
    try {
      assert.notEqual(leftover, pin);
      checkoutLlamaPin(dir, { llamacpp_commit: pin }, { fetch: false });
      assert.equal(llamaHead(dir), pin);
      assertLlamaAtPin(dir, { llamacpp_commit: pin });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("fail-closed when the pinned commit is absent from the leftover tree", () => {
    const { dir } = leftoverRepo();
    const missing = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    try {
      assert.throws(
        () => checkoutLlamaPin(dir, { llamacpp_commit: missing }, { fetch: false }),
        /not present in leftover tree/,
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("setup-reference.sh always checkouts pin.json (does not skip an existing tree)", () => {
    const src = readFileSync(join(HERE, "..", "scripts", "setup-reference.sh"), "utf8");
    assert.match(src, /ensure-llama-pin\.mjs" --checkout/);
    assert.doesNotMatch(src, /already cloned at/);
  });
});
