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

FEATURES="profile"
TARGET_DIR="$CRATE/target/profile-wasm-crate"
THREADS=0
if [[ "${MILTON_PROFILE_RELAXED:-}" == "1" ]]; then
  FEATURES="profile,relaxed-simd"
  TARGET_DIR="$CRATE/target/profile-wasm-relaxed-crate"
fi
if [[ "${MILTON_PROFILE_THREADS:-}" == "1" ]]; then
  THREADS=1
  FEATURES="${FEATURES},wasm-threads"
  if [[ "${MILTON_PROFILE_RELAXED:-}" == "1" ]]; then
    TARGET_DIR="$CRATE/target/profile-wasm-threads-relaxed-crate"
  else
    TARGET_DIR="$CRATE/target/profile-wasm-threads-crate"
  fi
fi

if [[ "$THREADS" == "1" ]]; then
  if ! rustup component list --installed | grep -q '^rust-src'; then
    rustup component add rust-src
  fi
  CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
  RUSTUP_HOME_DIR="${RUSTUP_HOME:-$HOME/.rustup}"
  REMAP="--remap-path-prefix=${CARGO_HOME_DIR}/registry/src=/cargo/registry/src --remap-path-prefix=${ROOT}=/milton --remap-path-prefix=${RUSTUP_HOME_DIR}=/rustup"
  export RUSTC_BOOTSTRAP=1
  export RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals ${REMAP}"
  cargo build --manifest-path "$CRATE/Cargo.toml" \
    --target wasm32-unknown-unknown --release --lib --features "$FEATURES" \
    --target-dir "$TARGET_DIR" \
    -Z build-std=std,panic_abort \
    -Z build-std-features=panic_immediate_abort
else
  cargo build --manifest-path "$CRATE/Cargo.toml" \
    --target wasm32-unknown-unknown --release --lib --features "$FEATURES" \
    --target-dir "$TARGET_DIR"
fi

RAW="$TARGET_DIR/wasm32-unknown-unknown/release/milton.wasm"
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
