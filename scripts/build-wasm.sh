#!/usr/bin/env bash
# Builder-side WASM-SIMD compile. Consumers do NOT run this.
# `npm i` uses the prebuilt wasm/milton_bg.wasm — no Rust, no node-gyp.
#
# Two artifacts (issue #44):
#   wasm/milton_bg.wasm          +simd128, ordinary path when SAB is absent
#   wasm/milton_threads_bg.wasm  +simd128,+atomics,+bulk-memory, shared memory
# Both remapped so panic locations are host-stable.
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
# rustup sysroot (std panic paths) is the leftover host-unstable prefix:
# this box /usr/local/rustup (17); GH ubuntu-latest /home/runner/.rustup (20).
# That 3-char delta was Data +24 on CI for milton_bg.wasm.
RUSTUP_HOME_DIR="${RUSTUP_HOME:-$HOME/.rustup}"
REMAP="--remap-path-prefix=${CARGO_HOME_DIR}/registry/src=/cargo/registry/src --remap-path-prefix=${ROOT}=/milton --remap-path-prefix=${RUSTUP_HOME_DIR}=/rustup"

# --- single-thread (ordinary path; no shared memory) ---
# Do not inherit +atomics from a prior threads build / leftover env.
unset RUSTC_BOOTSTRAP
export RUSTFLAGS="-C target-feature=+simd128 ${REMAP} ${MILTON_WASM_RUSTFLAGS:-}"

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

# --- threaded (shared memory). rust-src + bootstrap for build-std atomics. ---
# build-std crate-hash suffixes (hXXXX in the name section) still differ by
# rust-src host path after remap. CI is the build of record for
# milton_threads_bg.wasm — do not replace a CI-matching blob with a local
# rebuild unless CI byte-compare is red and you are committing CI's output.
if ! rustup component list --installed | grep -q '^rust-src'; then
  rustup component add rust-src
fi

# Do not leak single-thread rustflags / RUSTC_BOOTSTRAP into later native builds.
THREADS_RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals ${REMAP}"
(
  export RUSTC_BOOTSTRAP=1
  export RUSTFLAGS="${THREADS_RUSTFLAGS}"
  cargo build --manifest-path "$CRATE/Cargo.toml" \
    --target wasm32-unknown-unknown --release --lib --features wasm-threads \
    --target-dir "$CRATE/target/wasm-threads" \
    -Z build-std=std,panic_abort \
    -Z build-std-features=panic_immediate_abort
)

RAW_T="$CRATE/target/wasm-threads/wasm32-unknown-unknown/release/milton.wasm"
if [[ ! -f "$RAW_T" ]]; then
  echo "fail-closed: rustc did not emit $RAW_T" >&2
  exit 2
fi

wasm-bindgen "$RAW_T" \
  --out-dir "$OUT" \
  --target web \
  --out-name milton_threads \
  --omit-default-module-path
rm -f "$OUT/.gitignore"

WASM_T="$OUT/milton_threads_bg.wasm"
if [[ ! -f "$WASM_T" ]]; then
  echo "fail-closed: wasm-bindgen did not emit $WASM_T" >&2
  exit 2
fi

python3 - "$WASM" "$WASM_T" <<'PY'
import sys

def sections(path):
    data = open(path, "rb").read()
    if data[:4] != b"\0asm":
        print(f"fail-closed: {path} is not a wasm module", file=sys.stderr)
        sys.exit(1)
    i = 8
    def uvarint():
        nonlocal i
        x = 0
        s = 0
        while True:
            c = data[i]
            i += 1
            x |= (c & 0x7F) << s
            if c < 0x80:
                return x
            s += 7
    shared = False
    imported_mem = False
    while i < len(data):
        sid = uvarint()
        slen = uvarint()
        end = i + slen
        body = data[i:end]
        if sid == 2:
            j = 0
            def uv(b, k):
                x = 0
                s = 0
                while True:
                    c = b[k]
                    k += 1
                    x |= (c & 0x7F) << s
                    if c < 0x80:
                        return x, k
                    s += 7
            n, j = uv(body, 0)
            for _ in range(n):
                ml, j = uv(body, j)
                j += ml
                nl, j = uv(body, j)
                j += nl
                kind = body[j]
                j += 1
                if kind == 2:
                    flags = body[j]
                    imported_mem = True
                    shared = bool(flags & 2)
                    break
                break
        i = end
    fd = data.count(bytes([0xFD]))
    return data, fd, imported_mem, shared

single, s_fd, s_imp, s_sh = sections(sys.argv[1])
thr, t_fd, t_imp, t_sh = sections(sys.argv[2])
if s_fd == 0:
    print("fail-closed: milton_bg.wasm has no 0xfd SIMD opcodes", file=sys.stderr)
    sys.exit(1)
if t_fd == 0:
    print("fail-closed: milton_threads_bg.wasm has no 0xfd SIMD opcodes", file=sys.stderr)
    sys.exit(1)
if s_sh:
    print("fail-closed: milton_bg.wasm must not import shared memory", file=sys.stderr)
    sys.exit(1)
if not (t_imp and t_sh):
    print("fail-closed: milton_threads_bg.wasm must import shared memory", file=sys.stderr)
    sys.exit(1)
print(f"ok  {sys.argv[1]}  bytes={len(single)}  simd_fd_count={s_fd}  shared=0")
print(f"ok  {sys.argv[2]}  bytes={len(thr)}  simd_fd_count={t_fd}  shared=1")
PY
