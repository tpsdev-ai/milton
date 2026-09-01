#!/usr/bin/env bash
# CI guard (issue #16): the live setup-reference.sh compiler-resolution +
# cmake-arg construction must survive `set -u` when CC/CXX are unset.
# Isolates that block (cmake stubbed) — does not clone or compile llama.cpp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/harness/scripts/setup-reference.sh"

if [[ ! -f "$SRC" ]]; then
  echo "fail-closed: missing $SRC" >&2
  exit 1
fi

# Fresh-host repro: even if the caller exported compilers, drop them.
unset CC CXX

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
BLOCK_FILE="$WORKDIR/cmake-block.sh"
ARGS_FILE="$WORKDIR/cmake-args.txt"
BUGGY_ERR="$WORKDIR/buggy.err"

# Extract export/assignment lines immediately above `cmake -S` plus the
# configure continuation. A stale copy would not catch a regression.
node --input-type=module -e '
import { readFileSync } from "node:fs";
const src = readFileSync(process.argv[1], "utf8");
const lines = src.split("\n");
const cmakeIdx = lines.findIndex((l) => /\bcmake\s+-S\b/.test(l));
if (cmakeIdx < 0) {
  console.error("fail-closed: no cmake -S invocation in setup-reference.sh");
  process.exit(1);
}
let start = cmakeIdx;
while (start > 0 && /^(export\s+)?(CC|CXX)=/.test(lines[start - 1].trim())) {
  start -= 1;
}
let end = cmakeIdx;
while (end < lines.length && /\\\s*$/.test(lines[end])) {
  end += 1;
}
const block = lines.slice(start, end + 1).join("\n");
if (!block.trim()) {
  console.error("fail-closed: empty cmake block extracted from setup-reference.sh");
  process.exit(1);
}
process.stdout.write(block.endsWith("\n") ? block : `${block}\n`);
' "$SRC" > "$BLOCK_FILE"

# 1) The guard must be live: the historical prefix-assignment dies under set -u.
set +e
env -u CC -u CXX bash -c '
set -euo pipefail
cmake() { :; }
CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" cmake \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}"
' >/dev/null 2>"$BUGGY_ERR"
BUGGY_RC=$?
set -e
if [[ "$BUGGY_RC" -eq 0 ]] || ! grep -q "unbound variable" "$BUGGY_ERR"; then
  echo "fail-closed: historical CC prefix-assignment did not die under set -u — guard is inert" >&2
  cat "$BUGGY_ERR" >&2
  exit 1
fi

# 2) Run the live script's assignment + cmake-arg construction on a clean env.
env -u CC -u CXX LLAMA_DIR="$WORKDIR/llama" ARGS_FILE="$ARGS_FILE" BLOCK_FILE="$BLOCK_FILE" bash -c '
set -euo pipefail
mkdir -p "$LLAMA_DIR"
cmake() {
  printf "%s\n" "$@" > "$ARGS_FILE"
}
# shellcheck disable=SC1090
source "$BLOCK_FILE"
'

if [[ ! -s "$ARGS_FILE" ]]; then
  echo "fail-closed: stub cmake wrote no args — extracted block did not invoke cmake" >&2
  echo "extracted block:" >&2
  cat "$BLOCK_FILE" >&2
  exit 1
fi
if ! grep -q "CMAKE_C_COMPILER=/usr/bin/gcc" "$ARGS_FILE"; then
  echo "fail-closed: CMAKE_C_COMPILER did not default to /usr/bin/gcc" >&2
  cat "$ARGS_FILE" >&2
  exit 1
fi
if ! grep -q "CMAKE_CXX_COMPILER=/usr/bin/g++" "$ARGS_FILE"; then
  echo "fail-closed: CMAKE_CXX_COMPILER did not default to /usr/bin/g++" >&2
  cat "$ARGS_FILE" >&2
  exit 1
fi

echo "setup-cc-nounset: ok"
echo "  extracted $(wc -l < "$BLOCK_FILE") lines from setup-reference.sh"
echo "  historical prefix-assignment: unbound variable (guard live)"
echo "  clean-env cmake args: CMAKE_C_COMPILER=/usr/bin/gcc CMAKE_CXX_COMPILER=/usr/bin/g++"
