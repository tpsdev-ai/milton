#!/usr/bin/env bash
# Native milton-embed WITH --features rope-libm-sin.
# Separate --target-dir so the default release binary (no env-var read) is
# not overwritten. Used only by the wasm:compare must-fire receipt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/crate/target/rope-libm-sin"

cargo build --manifest-path "$ROOT/crate/Cargo.toml" \
  --release --bin milton-embed --features rope-libm-sin \
  --target-dir "$TARGET" >&2

BIN="$TARGET/release/milton-embed"
if [[ ! -f "$BIN" ]]; then
  echo "fail-closed: feature-enabled milton-embed missing at $BIN" >&2
  exit 2
fi
if ! grep -a -F -q 'MILTON_ROPE_LIBM_SIN' "$BIN"; then
  echo "fail-closed: --features rope-libm-sin binary has no MILTON_ROPE_LIBM_SIN string" >&2
  exit 2
fi
printf '%s\n' "$BIN"
