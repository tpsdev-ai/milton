#!/usr/bin/env python3
"""Per-function SIMD128 opcode coverage from wasm-objdump -d (issue #28 H4).

Opcode count of 0xfd in the whole file is not coverage. This attributes
SIMD instructions (v128 / *xN / 0xfd) to each function.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

FN_RE = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:")
SIMD_RE = re.compile(
    r"\b(v128|i8x16|i16x8|i32x4|i64x2|f32x4|f64x2|v8x16|v16x8)\b"
)


def analyze(path: str) -> dict:
    out = subprocess.check_output(["wasm-objdump", "-d", path], text=True, errors="replace")
    rows = []
    name = None
    n_ins = 0
    n_simd = 0
    simd_kinds: dict[str, int] = {}
    for line in out.splitlines():
        m = FN_RE.match(line.strip()) if line[:1] in "0123456789abcdefABCDEF" else None
        # wasm-objdump: "000123 func[12] <name>:"
        m = re.match(r"^[0-9a-fA-F]+\s+func\[\d+\]\s+<([^>]+)>:", line.strip())
        if m:
            if name is not None:
                rows.append(
                    {
                        "name": name,
                        "insns": n_ins,
                        "simd_insns": n_simd,
                        "simd_frac": (n_simd / n_ins) if n_ins else 0.0,
                        "simd_kinds": simd_kinds,
                    }
                )
            name = m.group(1)
            n_ins = 0
            n_simd = 0
            simd_kinds = {}
            continue
        if name is None:
            continue
        body = line.strip()
        if not body or body.startswith(";"):
            continue
        # instruction lines look like "  12: 20 00  local.get 0"
        if re.match(r"^[0-9a-fA-F]+:", body) or re.search(r"^\s+[0-9a-fA-F]+:", line):
            n_ins += 1
            kinds = SIMD_RE.findall(body)
            if kinds or "0xfd" in body.lower():
                n_simd += 1
                for k in kinds or ["0xfd"]:
                    simd_kinds[k] = simd_kinds.get(k, 0) + 1
    if name is not None:
        rows.append(
            {
                "name": name,
                "insns": n_ins,
                "simd_insns": n_simd,
                "simd_frac": (n_simd / n_ins) if n_ins else 0.0,
                "simd_kinds": simd_kinds,
            }
        )
    rows.sort(key=lambda r: (-r["simd_insns"], -r["insns"], r["name"]))
    with_simd = [r for r in rows if r["simd_insns"] > 0]
    return {
        "wasm": path,
        "n_functions": len(rows),
        "n_functions_with_simd": len(with_simd),
        "simd_insn_total": sum(r["simd_insns"] for r in rows),
        "functions": rows,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wasm-simd-per-fn.py milton_bg.wasm", file=sys.stderr)
        return 2
    report = analyze(sys.argv[1])
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
