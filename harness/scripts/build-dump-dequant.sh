#!/usr/bin/env bash
# Build the llama.cpp dequant oracle dumper. Harness-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="$ROOT/harness/vendor/llama.cpp"
BIN="$LLAMA/build/bin"
OUT="$ROOT/harness/vendor/bin/dump-dequant"
mkdir -p "$(dirname "$OUT")"
if [[ ! -x "$BIN/llama-embedding" ]]; then
  echo "fail-closed: llama-embedding missing — run npm run harness:setup" >&2
  exit 1
fi
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
"$CXX" -std=c++17 -O2 \
  -I"$LLAMA/ggml/include" \
  -L"$BIN" -Wl,-rpath,"$BIN" \
  -o "$OUT" \
  "$ROOT/harness/tools/dump-dequant.cpp" \
  -lggml -lggml-base -lggml-cpu
echo "dump-dequant: $OUT"
