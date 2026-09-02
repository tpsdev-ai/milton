#!/usr/bin/env python3
"""Emit CI pins from repo files. Versions are not duplicated in YAML.

Writes GitHub Actions `name=value` lines to stdout (and to $GITHUB_OUTPUT
when that env var is set). Fail-closed on a missing or malformed pin.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"fail-closed: {msg}", file=sys.stderr)
    sys.exit(2)


def read_rust_toolchain() -> tuple[str, str]:
    text = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    channel_m = re.search(r'(?m)^channel\s*=\s*"([^"]+)"\s*$', text)
    if not channel_m:
        fail("rust-toolchain.toml missing channel")
    channel = channel_m.group(1).strip()
    if not channel:
        fail("rust-toolchain.toml channel is empty")
    targets_m = re.search(r"targets\s*=\s*\[(.*?)\]", text, re.S)
    if not targets_m:
        fail("rust-toolchain.toml missing targets")
    targets = re.findall(r'"([^"]+)"', targets_m.group(1))
    if "wasm32-unknown-unknown" not in targets:
        fail("rust-toolchain.toml must list wasm32-unknown-unknown")
    return channel, ",".join(targets)


def read_node_major() -> str:
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    engines = pkg.get("engines") or {}
    raw = engines.get("node")
    if not isinstance(raw, str) or not raw.strip():
        fail("package.json engines.node missing")
    m = re.search(r"(\d+)", raw)
    if not m:
        fail(f"package.json engines.node has no major version: {raw!r}")
    return m.group(1)


def read_bindgen_version() -> str:
    text = (ROOT / "scripts" / "build-wasm.sh").read_text(encoding="utf-8")
    m = re.search(r'(?m)^BINDGEN_VERSION="([^"]+)"\s*$', text)
    if not m or not m.group(1).strip():
        fail("scripts/build-wasm.sh missing BINDGEN_VERSION")
    return m.group(1).strip()


def read_q4_pin() -> tuple[str, str, str, str]:
    pin = json.loads((ROOT / "harness" / "goldens" / "pin.json").read_text(encoding="utf-8"))
    sha = pin.get("gguf_sha256")
    name = pin.get("gguf_file")
    source = pin.get("gguf_source")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        fail("pin.json gguf_sha256 must be a 64-char lowercase hex digest")
    if not isinstance(name, str) or not name.endswith(".gguf"):
        fail("pin.json gguf_file missing")
    if not isinstance(source, str) or not source.startswith("https://"):
        fail("pin.json gguf_source missing")
    url = f"{source.rstrip('/')}/resolve/main/{name}"
    return sha, name, source, url


def main() -> int:
    channel, targets = read_rust_toolchain()
    node_major = read_node_major()
    bindgen = read_bindgen_version()
    sha, name, source, url = read_q4_pin()

    rows = {
        "rust_channel": channel,
        "rust_targets": targets,
        "node_major": node_major,
        "bindgen_version": bindgen,
        "gguf_sha256": sha,
        "gguf_file": name,
        "gguf_source": source,
        "gguf_url": url,
        "gguf_relpath": f"harness/models/{name}",
    }
    out_path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in rows.items()]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
