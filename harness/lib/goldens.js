import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { sha256json } from "./digest.js";

const HERE = dirname(fileURLToPath(import.meta.url));
export const GOLDENS_PATH = join(HERE, "..", "goldens", "vectors.json");
export const PIN_PATH = join(HERE, "..", "goldens", "pin.json");
export const EPSILON_PATH = join(HERE, "..", "goldens", "epsilon.json");

export function loadGoldens(path = GOLDENS_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (!raw || raw.schema !== "milton.goldens/1" || !Array.isArray(raw.items)) {
    throw new Error(`invalid goldens at ${path}: expected schema milton.goldens/1`);
  }
  return raw;
}

export function loadPin(path = PIN_PATH) {
  return JSON.parse(readFileSync(path, "utf8"));
}

export function loadEpsilon(path = EPSILON_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (typeof raw.epsilon !== "number" || typeof raw.epsilon_abs !== "number") {
    throw new Error(`invalid epsilon file at ${path}`);
  }
  if (!(raw.epsilon > 0) || !(raw.epsilon_abs > 0)) {
    throw new Error(`epsilon values must be positive (got ${raw.epsilon} / ${raw.epsilon_abs})`);
  }
  return raw;
}

export function goldensById(goldens) {
  const map = new Map();
  for (const item of goldens.items) {
    map.set(item.id, item);
  }
  return map;
}

/** Content-address of the pinned vectors (ids + float bytes), not the prose. */
export function referenceDigest(goldens) {
  return sha256json(
    goldens.items.map((item) => ({
      id: item.id,
      prefix: item.prefix,
      dims: item.dims,
      vector: item.vector,
    })),
  );
}

export function asFloat32(vector) {
  return Float32Array.from(vector);
}
