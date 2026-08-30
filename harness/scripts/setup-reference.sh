#!/usr/bin/env bash
# Fetch the pinned GGUF and build llama-embedding. Harness-only — never src/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENDOR="$ROOT/harness/vendor"
MODEL_DIR="$VENDOR/models"
LLAMA_DIR="$VENDOR/llama.cpp"
GGUF_NAME="nomic-embed-text-v1.5.Q4_K_M.gguf"
GGUF_SHA256="d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac"
# Pinned llama.cpp commit (recorded again in harness/goldens/pin.json after generate).
# b6399 — embedding tool + mean pooling + --embd-normalize. Recorded again in pin.json.
LLAMA_COMMIT="${MILTON_LLAMA_COMMIT:-b6399}"
GGUF_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/${GGUF_NAME}"

mkdir -p "$MODEL_DIR"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "cloning llama.cpp @ ${LLAMA_COMMIT}..."
  git clone --depth 1 --branch "$LLAMA_COMMIT" https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" \
    || git clone --depth 1 --branch "$LLAMA_COMMIT" https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
  echo "llama.cpp already cloned at $LLAMA_DIR"
fi

echo "building llama-embedding (CPU, gcc/g++)..."
# Force gcc: some images have /usr/bin/c++ → clang, which then fails to find -lstdc++.
rm -rf "$LLAMA_DIR/build"
CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_CCACHE=OFF
cmake --build "$LLAMA_DIR/build" --target llama-embedding -j"$(nproc)"

BIN="$LLAMA_DIR/build/bin/llama-embedding"
if [[ ! -x "$BIN" ]]; then
  # some layouts put it under build/bin/Release
  BIN="$(find "$LLAMA_DIR/build" -name llama-embedding -type f | head -n1)"
fi
test -x "$BIN"
echo "llama-embedding: $BIN"
"$BIN" --version 2>/dev/null || true
git -C "$LLAMA_DIR" rev-parse HEAD
git -C "$LLAMA_DIR" rev-parse HEAD | sha256sum | awk '{print $1}'

GGUF="$MODEL_DIR/$GGUF_NAME"
if [[ -f "$GGUF" ]]; then
  GOT="$(sha256sum "$GGUF" | awk '{print $1}')"
  if [[ "$GOT" != "$GGUF_SHA256" ]]; then
    echo "stale GGUF digest $GOT — re-fetching"
    rm -f "$GGUF"
  fi
fi
if [[ ! -f "$GGUF" ]]; then
  echo "fetching $GGUF_NAME ..."
  curl -L --fail --retry 4 --retry-delay 4 -o "$GGUF" "$GGUF_URL"
fi
GOT="$(sha256sum "$GGUF" | awk '{print $1}')"
if [[ "$GOT" != "$GGUF_SHA256" ]]; then
  echo "GGUF digest mismatch: got $GOT want $GGUF_SHA256" >&2
  exit 1
fi
echo "GGUF ok  sha256=$GOT  bytes=$(wc -c < "$GGUF")"
echo "setup-reference: done"
