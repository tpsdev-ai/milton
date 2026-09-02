#!/usr/bin/env bash
# Profile-feature WASM. Writes to a separate dir — never overwrites wasm/.
# Default wasm:build / committed milton_bg.wasm stay bit-identical.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/crate"
OUT="${1:-$ROOT/crate/target/profile-wasm}"
BINDGEN_VERSION="0.2.100"

if ! command -v rustup >/dev/null 2>&1; then
  echo "fail-closed: rustup is required" >&2
  exit 2
fi
rustup target add wasm32-unknown-unknown >/dev/null

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "fail-closed: wasm-bindgen CLI ${BINDGEN_VERSION} is required" >&2
  exit 2
fi
GOT="$(wasm-bindgen --version | awk '{print $2}')"
if [[ "$GOT" != "$BINDGEN_VERSION" ]]; then
  echo "fail-closed: wasm-bindgen CLI is $GOT, crate pins ${BINDGEN_VERSION}" >&2
  exit 2
fi

CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128 --remap-path-prefix=${CARGO_HOME_DIR}/registry/src=/cargo/registry/src --remap-path-prefix=${ROOT}=/milton"

cargo build --manifest-path "$CRATE/Cargo.toml" \
  --target wasm32-unknown-unknown --release --lib --features profile \
  --target-dir "$CRATE/target/profile-wasm-crate"

RAW="$CRATE/target/profile-wasm-crate/wasm32-unknown-unknown/release/milton.wasm"
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

rm -f "$OUT/.gitignore"
echo "ok  profile wasm -> $OUT/milton_bg.wasm  bytes=$(wc -c < "$OUT/milton_bg.wasm")"
