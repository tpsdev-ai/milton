#!/usr/bin/env bash
# Section-table diff of committed vs rebuilt milton_bg.wasm (Flint #30 ASK 2).
# Fail-closed if wabt is missing. Does not change the byte-compare verdict.
set -euo pipefail

committed="${1:?committed wasm path}"
rebuilt="${2:?rebuilt wasm path}"

if ! command -v wasm-objdump >/dev/null 2>&1; then
  echo "fail-closed: wasm-objdump (wabt) is required for the section dump" >&2
  exit 2
fi

echo "======== committed wasm-objdump -h ========"
wasm-objdump -h "$committed"
echo "======== rebuilt wasm-objdump -h ========"
wasm-objdump -h "$rebuilt"

python3 - "$committed" "$rebuilt" <<'PY'
import subprocess, sys

def sections(path):
    out = subprocess.check_output(["wasm-objdump", "-h", path], text=True)
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            name = parts[1]
            size = int(parts[2], 16)
            # custom sections include a quoted name in later columns
            if name == "Custom" and '"' in line:
                q = line.split('"')
                if len(q) >= 2:
                    name = f'Custom:{q[1]}'
            rows[name] = size
    return rows

committed, rebuilt = sys.argv[1], sys.argv[2]
c = sections(committed)
r = sections(rebuilt)
keys = sorted(set(c) | set(r))
print("======== section size delta (rebuilt - committed) ========")
print(f"{'section':28} {'committed':>12} {'rebuilt':>12} {'delta':>8}")
differ = []
for k in keys:
    cs, rs = c.get(k), r.get(k)
    if cs is None:
        print(f"{k:28} {'-':>12} {rs:12}      NEW")
        differ.append(k)
    elif rs is None:
        print(f"{k:28} {cs:12} {'-':>12}  DROPPED")
        differ.append(k)
    else:
        d = rs - cs
        mark = "  <<<" if d else ""
        print(f"{k:28} {cs:12} {rs:12} {d:+8}{mark}")
        if d:
            differ.append(k)
if not differ:
    print("no section-size delta")
    raise SystemExit(0)
print("differing_sections=" + ",".join(differ))
open("/tmp/milton-wasm-differing-sections", "w").write("\n".join(differ) + "\n")
PY

if [[ -f /tmp/milton-wasm-differing-sections ]]; then
  while IFS= read -r sec; do
    [[ -z "$sec" ]] && continue
    # Custom:name -> -j name; otherwise -j section
    j="${sec#Custom:}"
    echo "======== committed wasm-objdump -x -j ${j} ========"
    wasm-objdump -x -j "$j" "$committed" || true
    echo "======== rebuilt wasm-objdump -x -j ${j} ========"
    wasm-objdump -x -j "$j" "$rebuilt" || true
  done < /tmp/milton-wasm-differing-sections
fi
