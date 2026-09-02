#!/usr/bin/env bash
# Builder-side WASM-SIMD compile. Consumers do NOT run this.
# `npm i` uses the prebuilt wasm/milton_bg.wasm — no Rust, no node-gyp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/crate"
OUT="$ROOT/wasm"
BINDGEN_VERSION="0.2.100"

if ! command -v rustup >/dev/null 2>&1; then
  echo "fail-closed: rustup is required to produce the prebuilt wasm (builder-side only)" >&2
  exit 2
fi
rustup target add wasm32-unknown-unknown >/dev/null

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "fail-closed: wasm-bindgen CLI ${BINDGEN_VERSION} is required (builder-side). Install with:" >&2
  echo "  cargo install wasm-bindgen-cli --version ${BINDGEN_VERSION} --locked" >&2
  exit 2
fi
GOT="$(wasm-bindgen --version | awk '{print $2}')"
if [[ "$GOT" != "$BINDGEN_VERSION" ]]; then
  echo "fail-closed: wasm-bindgen CLI is $GOT, crate pins ${BINDGEN_VERSION}" >&2
  exit 2
fi

# SIMD128 — also in crate/.cargo/config.toml.
# Remap CARGO_HOME registry + repo root so panic locations are host-stable.
# Builder box is /usr/local/cargo (16 chars); GH ubuntu-latest is
# /home/runner/.cargo (19). That 3-char prefix delta was Data +8 on CI.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128 --remap-path-prefix=${CARGO_HOME_DIR}/registry/src=/cargo/registry/src --remap-path-prefix=${ROOT}=/milton"

cargo build --manifest-path "$CRATE/Cargo.toml" \
  --target wasm32-unknown-unknown --release --lib

RAW="$CRATE/target/wasm32-unknown-unknown/release/milton.wasm"
if [[ ! -f "$RAW" ]]; then
  echo "fail-closed: rustc did not emit $RAW" >&2
  exit 2
fi

mkdir -p "$OUT"
wasm-bindgen "$RAW" \
  --out-dir "$OUT" \
  --target web \
  --out-name milton \
  --omit-default-module-path

# wasm-bindgen --target web writes a .gitignore that ignores the generated
# files. We ship those files; drop it so git will track the prebuilt wasm.
rm -f "$OUT/.gitignore"

WASM="$OUT/milton_bg.wasm"
if [[ ! -f "$WASM" ]]; then
  echo "fail-closed: wasm-bindgen did not emit $WASM" >&2
  exit 2
fi

python3 - "$WASM" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
n = data.count(bytes([0xFD]))
if n == 0:
    print("fail-closed: milton_bg.wasm has no 0xfd SIMD opcodes", file=sys.stderr)
    sys.exit(1)
print(f"ok  {sys.argv[1]}  bytes={len(data)}  simd_fd_count={n}")
PY
