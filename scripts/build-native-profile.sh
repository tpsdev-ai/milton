#!/usr/bin/env bash
# Native milton-profile WITH --features profile.
# Separate --target-dir so the default release binary is not overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/crate/target/profile-native"

cargo build --manifest-path "$ROOT/crate/Cargo.toml" \
  --release --bin milton-profile --features profile \
  --target-dir "$TARGET" >&2

BIN="$TARGET/release/milton-profile"
if [[ ! -f "$BIN" ]]; then
  echo "fail-closed: feature-enabled milton-profile missing at $BIN" >&2
  exit 2
fi
printf '%s\n' "$BIN"
