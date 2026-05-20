import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { initAgent } from "../src/init.js";
import { mkdtempSync, rmSync, readFileSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

describe("initAgent", () => {
  let tmpRoot: string;
  let keysRoot: string;

  beforeEach(() => {
    tmpRoot = mkdtempSync(join(tmpdir(), "milton-init-"));
    keysRoot = mkdtempSync(join(tmpdir(), "milton-keys-"));
  });

  afterEach(() => {
    rmSync(tmpRoot, { recursive: true, force: true });
    rmSync(keysRoot, { recursive: true, force: true });
  });

  // Shared default opts — every test routes Flair keys into a temp dir
  // so we never touch ~/.flair/keys/. Tests that need to vary something
  // override via spread.
  const baseOpts = () => ({
    name: "testbot",
    role: "ea" as const,
    provider: "ollama-cloud",
    model: "kimi-k2.6",
    agentsRoot: tmpRoot,
    flairKeysDir: keysRoot,
  });

  it("creates the full directory tree", () => {
    const res = initAgent(baseOpts());
    expect(res.agentDir).toBe(join(tmpRoot, "testbot"));
    expect(existsSync(join(res.agentDir, "bin"))).toBe(true);
    expect(existsSync(join(res.agentDir, "work"))).toBe(true);
    expect(existsSync(join(res.agentDir, "memory"))).toBe(true);
    expect(existsSync(join(res.agentDir, ".pi-agent"))).toBe(true);
  });

  it("writes soul.md from the role template", () => {
    const res = initAgent(baseOpts());
    const soul = readFileSync(join(res.agentDir, "soul.md"), "utf8");
    expect(soul).toContain("EA");
    expect(soul.length).toBeGreaterThan(500);
  });

  it("writes milton.yaml with provider + model + tools", () => {
    const res = initAgent({ ...baseOpts(), provider: "exe-dev-gateway", model: "claude-opus-4-7" });
    const yaml = readFileSync(join(res.agentDir, "milton.yaml"), "utf8");
    expect(yaml).toContain("id: testbot");
    expect(yaml).toContain("role: ea");
    expect(yaml).toContain("name: exe-dev-gateway");
    expect(yaml).toContain("model: claude-opus-4-7");
    expect(yaml).toContain("- Bash");
    expect(yaml).toContain("- mcp__flair__memory_store");
  });

  it("generates an executable bin/<name> launcher", () => {
    const res = initAgent(baseOpts());
    const bin = join(res.agentDir, "bin", "testbot");
    const launcher = readFileSync(bin, "utf8");
    expect(launcher).toContain("#!/bin/sh");
    expect(launcher).toContain("pi --provider ollama-cloud --model kimi-k2.6");
    expect(launcher).toContain("PI_CODING_AGENT_DIR=");
    const mode = statSync(bin).mode & 0o777;
    expect(mode & 0o111).toBeGreaterThan(0);
  });

  it("uses sh shebang (not zsh) — preserves env-prefix invocations", () => {
    const res = initAgent(baseOpts());
    const launcher = readFileSync(join(res.agentDir, "bin", "testbot"), "utf8");
    expect(launcher.startsWith("#!/bin/sh")).toBe(true);
    expect(launcher.startsWith("#!/bin/zsh")).toBe(false);
  });

  it("refuses to clobber an existing agent dir by default", () => {
    const opts = baseOpts();
    initAgent(opts);
    expect(() => initAgent(opts)).toThrow(/already exists/);
  });

  it("overwrites when noClobber=false", () => {
    const opts = baseOpts();
    initAgent(opts);
    const res = initAgent({ ...opts, noClobber: false });
    expect(res.files.length).toBeGreaterThan(0);
  });

  it("rejects invalid agent names (path-traversal guard)", () => {
    expect(() => initAgent({ ...baseOpts(), name: "../etc" })).toThrow(/invalid agent name/);
  });

  it("propagates loadRole's invalid-role rejection", () => {
    expect(() => initAgent({
      ...baseOpts(),
      // @ts-expect-error — bad role
      role: "../../../etc",
    })).toThrow(/invalid role name/);
  });

  it("generates a Flair Ed25519 keypair by default", () => {
    const res = initAgent(baseOpts());
    expect(res.flair).toBeDefined();
    expect(res.flair?.publicKeyBase64.length).toBe(44);
    expect(existsSync(res.flair!.privateKeyPath)).toBe(true);
    expect(existsSync(res.flair!.publicKeyPath)).toBe(true);
    expect(res.files).toContain(res.flair!.privateKeyPath);
    expect(res.files).toContain(res.flair!.publicKeyPath);
  });

  it("skips Flair keypair when skipFlair=true", () => {
    const res = initAgent({ ...baseOpts(), skipFlair: true });
    expect(res.flair).toBeUndefined();
    // Files list shouldn't include any .key/.pub paths
    expect(res.files.some((f) => f.endsWith(".key"))).toBe(false);
    expect(res.files.some((f) => f.endsWith(".pub"))).toBe(false);
  });

  it("private key is mode 0600", () => {
    const res = initAgent(baseOpts());
    const mode = statSync(res.flair!.privateKeyPath).mode & 0o777;
    expect(mode).toBe(0o600);
  });
});
