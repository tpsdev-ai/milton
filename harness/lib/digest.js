import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

/** Canonical JSON: sorted keys, no whitespace. Arrays keep order. */
export function canonicalJson(value) {
  return JSON.stringify(sortDeep(value));
}

function sortDeep(v) {
  if (Array.isArray(v)) return v.map(sortDeep);
  if (v && typeof v === "object") {
    const out = {};
    for (const k of Object.keys(v).sort()) out[k] = sortDeep(v[k]);
    return out;
  }
  return v;
}

export function sha256hex(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

export function sha256json(value) {
  return sha256hex(canonicalJson(value));
}

export function sha256file(path) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}
