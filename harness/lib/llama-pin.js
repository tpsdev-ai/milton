/**
 * Fail-closed pin for the llama.cpp tree used as the reference oracle.
 *
 * pin.json's `llamacpp_commit` is the only acceptable HEAD. A leftover
 * vendor/llama.cpp clone must not be rebuilt in place.
 */
import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

const FULL_SHA = /^[0-9a-f]{40}$/;

export function pinnedLlamaCommit(pin) {
  const c = pin?.llamacpp_commit;
  if (typeof c !== "string" || !FULL_SHA.test(c)) {
    throw new Error(
      `fail-closed: pin.json llamacpp_commit must be a full 40-char sha, got ${JSON.stringify(c)}`,
    );
  }
  return c.toLowerCase();
}

export function llamaHead(llamaDir) {
  try {
    return execFileSync("git", ["-C", llamaDir, "rev-parse", "HEAD"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    })
      .trim()
      .toLowerCase();
  } catch (err) {
    throw new Error(`fail-closed: cannot read HEAD in ${llamaDir}: ${err.message}`);
  }
}

export function commitExists(llamaDir, sha) {
  try {
    execFileSync("git", ["-C", llamaDir, "cat-file", "-e", `${sha}^{commit}`], {
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

export function assertLlamaAtPin(llamaDir, pin) {
  const want = pinnedLlamaCommit(pin);
  if (!existsSync(join(llamaDir, ".git"))) {
    throw new Error(`fail-closed: no llama.cpp git tree at ${llamaDir}`);
  }
  const got = llamaHead(llamaDir);
  if (got !== want) {
    throw new Error(`fail-closed: leftover llama.cpp tree HEAD ${got} != pin.json ${want}`);
  }
  return got;
}

/**
 * Check out pin.json's commit. Fail closed if the tree is missing, the
 * commit cannot be fetched, or HEAD still does not match after checkout.
 */
export function checkoutLlamaPin(llamaDir, pin, { fetch = true } = {}) {
  const want = pinnedLlamaCommit(pin);
  if (!existsSync(join(llamaDir, ".git"))) {
    throw new Error(`fail-closed: no llama.cpp git tree at ${llamaDir} — clone first, then checkout the pin`);
  }
  if (!commitExists(llamaDir, want)) {
    if (!fetch) {
      throw new Error(`fail-closed: pinned commit ${want} not present in leftover tree ${llamaDir}`);
    }
    try {
      execFileSync("git", ["-C", llamaDir, "fetch", "--depth", "1", "origin", want], {
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch {
      try {
        execFileSync("git", ["-C", llamaDir, "fetch", "origin", want], {
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (err) {
        throw new Error(`fail-closed: cannot fetch pinned llama.cpp ${want}: ${err.message}`);
      }
    }
    if (!commitExists(llamaDir, want)) {
      throw new Error(`fail-closed: pinned commit ${want} still absent after fetch`);
    }
  }
  execFileSync("git", ["-C", llamaDir, "checkout", "--detach", want], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  return assertLlamaAtPin(llamaDir, pin);
}
