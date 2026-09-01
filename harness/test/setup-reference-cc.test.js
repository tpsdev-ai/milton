import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const SETUP = join(HERE, "..", "scripts", "setup-reference.sh");
const GUARD = join(HERE, "..", "scripts", "check-setup-cc-nounset.sh");

function runCleanEnv(args, opts = {}) {
  return execFileSync("env", ["-u", "CC", "-u", "CXX", ...args], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    cwd: ROOT,
    ...opts,
  });
}

describe("setup-reference.sh CC/CXX under set -u (issue #16)", () => {
  it("the historical prefix-assignment dies on a clean env (guard is live)", () => {
    let err;
    try {
      runCleanEnv([
        "bash",
        "-c",
        [
          "set -euo pipefail",
          "cmake() { :; }",
          'CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" cmake \\',
          '  -DCMAKE_C_COMPILER="${CC}" \\',
          '  -DCMAKE_CXX_COMPILER="${CXX}"',
        ].join("\n"),
      ]);
    } catch (e) {
      err = e;
    }
    assert.ok(err, "expected nounset abort");
    assert.match(String(err.stderr || err.message), /unbound variable/);
  });

  it("setup-reference.sh exports CC/CXX before unguarded \${CC}/\${CXX} expansions", () => {
    const src = readFileSync(SETUP, "utf8");
    const exportCC = src.search(/^export CC=/m);
    const exportCXX = src.search(/^export CXX=/m);
    const cmakeC = src.indexOf('-DCMAKE_C_COMPILER="${CC}"');
    const cmakeCXX = src.indexOf('-DCMAKE_CXX_COMPILER="${CXX}"');
    assert.ok(exportCC >= 0, "missing export CC=");
    assert.ok(exportCXX >= 0, "missing export CXX=");
    assert.ok(cmakeC >= 0, "missing -DCMAKE_C_COMPILER");
    assert.ok(cmakeCXX >= 0, "missing -DCMAKE_CXX_COMPILER");
    assert.ok(exportCC < cmakeC, "export CC must precede -DCMAKE_C_COMPILER");
    assert.ok(exportCXX < cmakeCXX, "export CXX must precede -DCMAKE_CXX_COMPILER");
    assert.doesNotMatch(
      src,
      /^CC="\$\{CC:-[^}]+\}"\s+CXX="\$\{CXX:-[^}]+\}"\s+cmake/m,
      "prefix-assignment cmake must not return",
    );
  });

  it("env -u CC -u CXX bash harness/scripts/check-setup-cc-nounset.sh", () => {
    const out = runCleanEnv(["bash", GUARD]);
    assert.match(out, /setup-cc-nounset: ok/);
    assert.match(out, /guard live/);
    assert.match(out, /CMAKE_C_COMPILER=\/usr\/bin\/gcc/);
    assert.match(out, /CMAKE_CXX_COMPILER=\/usr\/bin\/g\+\+/);
  });
});
