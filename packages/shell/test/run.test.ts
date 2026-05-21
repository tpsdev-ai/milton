import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { EventEmitter } from "node:events";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { SpawnFn } from "../src/onboard.js";
import { runAgent } from "../src/run.js";

function fakeSpawn(opts: {
  exitCode?: number;
  stdoutChunks?: string[];
  onSpawn?: (cmd: string, args: readonly string[]) => void;
}): SpawnFn {
  return (cmd, args) => {
    const proc = new EventEmitter() as EventEmitter & {
      stdout?: EventEmitter & { setEncoding?: (enc: string) => void };
    };
    if (opts.stdoutChunks) {
      const stdoutStream = new EventEmitter() as EventEmitter & {
        setEncoding?: (enc: string) => void;
      };
      stdoutStream.setEncoding = () => {};
      proc.stdout = stdoutStream;
    }
    opts.onSpawn?.(cmd, args as readonly string[]);
    queueMicrotask(() => {
      if (opts.stdoutChunks && proc.stdout) {
        for (const chunk of opts.stdoutChunks) {
          proc.stdout.emit("data", chunk);
        }
      }
      proc.emit("exit", opts.exitCode ?? 0);
    });
    // biome-ignore lint/suspicious/noExplicitAny: minimal ChildProcess stub
    return proc as any;
  };
}

describe("runAgent", () => {
  let agentsRoot: string;

  beforeEach(() => {
    agentsRoot = mkdtempSync(join(tmpdir(), "bob-run-"));
    // Create a minimal launcher at <agentsRoot>/testbot/bin/testbot
    const launcherDir = join(agentsRoot, "testbot", "bin");
    mkdirSync(launcherDir, { recursive: true });
    const launcherPath = join(launcherDir, "testbot");
    writeFileSync(launcherPath, "#!/bin/sh\necho ok\n");
    chmodSync(launcherPath, 0o755);
  });

  afterEach(() => {
    rmSync(agentsRoot, { recursive: true, force: true });
  });

  it("rejects invalid agent names (path-traversal defense)", async () => {
    await expect(
      runAgent({ name: "../../etc", agentsRoot, spawnFn: fakeSpawn({}) }),
    ).rejects.toThrow(/invalid agent name/);
  });

  it("errors when the launcher does not exist", async () => {
    await expect(
      runAgent({ name: "missingbot", agentsRoot, spawnFn: fakeSpawn({}) }),
    ).rejects.toThrow(/launcher not found/);
  });

  it("spawns the launcher with no extra args when invoked bare", async () => {
    let captured: { cmd: string; args: readonly string[] } | undefined;
    const spawnFn = fakeSpawn({
      onSpawn: (cmd, args) => {
        captured = { cmd, args };
      },
    });
    const res = await runAgent({ name: "testbot", agentsRoot, spawnFn });
    expect(res.exitCode).toBe(0);
    expect(captured?.cmd).toBe(join(agentsRoot, "testbot", "bin", "testbot"));
    expect(captured?.args).toEqual([]);
  });

  it("passes --model when model override is set", async () => {
    let captured: readonly string[] = [];
    const spawnFn = fakeSpawn({
      onSpawn: (_cmd, args) => {
        captured = args;
      },
    });
    await runAgent({ name: "testbot", model: "claude-sonnet-4-6", agentsRoot, spawnFn });
    expect(captured).toContain("--model");
    expect(captured[captured.indexOf("--model") + 1]).toBe("claude-sonnet-4-6");
  });

  it("passes --print + prompt when prompt is provided (non-interactive default)", async () => {
    let captured: readonly string[] = [];
    const spawnFn = fakeSpawn({
      onSpawn: (_cmd, args) => {
        captured = args;
      },
    });
    await runAgent({ name: "testbot", prompt: "hello", agentsRoot, spawnFn });
    expect(captured).toContain("--print");
    expect(captured).toContain("hello");
  });

  it("omits --print when interactive=true", async () => {
    let captured: readonly string[] = [];
    const spawnFn = fakeSpawn({
      onSpawn: (_cmd, args) => {
        captured = args;
      },
    });
    await runAgent({
      name: "testbot",
      prompt: "hello",
      interactive: true,
      agentsRoot,
      spawnFn,
    });
    expect(captured).not.toContain("--print");
    expect(captured).toContain("hello");
  });

  it("combines --model + prompt correctly", async () => {
    let captured: readonly string[] = [];
    const spawnFn = fakeSpawn({
      onSpawn: (_cmd, args) => {
        captured = args;
      },
    });
    await runAgent({
      name: "testbot",
      prompt: "draft the brief",
      model: "claude-opus-4-7",
      agentsRoot,
      spawnFn,
    });
    // Order: --model X --print prompt
    expect(captured).toEqual(["--model", "claude-opus-4-7", "--print", "draft the brief"]);
  });

  it("propagates non-zero exit codes from the launcher", async () => {
    const spawnFn = fakeSpawn({ exitCode: 42 });
    const res = await runAgent({ name: "testbot", agentsRoot, spawnFn });
    expect(res.exitCode).toBe(42);
  });

  describe("captureStdout", () => {
    it("captures stdout when captureStdout=true", async () => {
      const spawnFn = fakeSpawn({
        stdoutChunks: ["partial reply ", "more reply\n"],
      });
      const res = await runAgent({
        name: "testbot",
        prompt: "hi",
        captureStdout: true,
        agentsRoot,
        spawnFn,
      });
      expect(res.stdout).toBe("partial reply more reply\n");
    });

    it("omits stdout from result when captureStdout=false (default)", async () => {
      const spawnFn = fakeSpawn({});
      const res = await runAgent({ name: "testbot", prompt: "hi", agentsRoot, spawnFn });
      expect(res.stdout).toBeUndefined();
    });

    it("rejects captureStdout combined with interactive (incompatible)", async () => {
      await expect(
        runAgent({
          name: "testbot",
          prompt: "hi",
          captureStdout: true,
          interactive: true,
          agentsRoot,
          spawnFn: fakeSpawn({}),
        }),
      ).rejects.toThrow(/captureStdout is incompatible with interactive/);
    });
  });
});
