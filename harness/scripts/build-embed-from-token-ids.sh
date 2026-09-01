#!/usr/bin/env bash
# Build the llama.cpp token-ID embedding oracle. Harness-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="$ROOT/harness/vendor/llama.cpp"
BIN="$LLAMA/build/bin"
OUT="$ROOT/harness/vendor/bin/embed-from-token-ids"
mkdir -p "$(dirname "$OUT")"
if [[ ! -x "$BIN/llama-embedding" ]]; then
  echo "fail-closed: llama-embedding missing — run npm run harness:setup" >&2
  exit 1
fi
if [[ ! -f "$LLAMA/include/llama.h" ]]; then
  echo "fail-closed: pinned llama.cpp headers missing at $LLAMA/include/llama.h" >&2
  exit 1
fi
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
"$CXX" -std=c++17 -O2 \
  -I"$LLAMA/include" \
  -I"$LLAMA/ggml/include" \
  -L"$BIN" -Wl,-rpath,"$BIN" \
  -o "$OUT" \
  "$ROOT/harness/tools/embed-from-token-ids.cpp" \
  -lllama -lggml -lggml-base -lggml-cpu -pthread
echo "embed-from-token-ids: $OUT"
