// `bob run <name> [prompt]` — invoke an onboarded agent's launcher.
//
// PR-1 shipped this as a stub. PR-16b makes it real: spawn the agent's
// generated bin/<name> launcher with optional model override and prompt.
//
// Model override is per-call. The default (in bob.yaml) is what the
// launcher hardcodes via `pi --model <X>`; passing `--model` to the
// launcher overrides it because pi treats later flags as authoritative.
//
// A full "task-based routes" table in bob.yaml is intentionally NOT in
// this PR — that needs a YAML parser dependency we haven't added yet,
// and the explicit `--model X` override already covers the per-call
// dynamic-routing use case (cron entries embed the right model in the
// command, conversational users pass --model).

import { type ChildProcess, spawn as nodeSpawn, type SpawnOptions } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { SpawnFn } from "./onboard.js";

// Same regex as init.ts AGENT_NAME — names are filesystem paths, keep them
// strict-safe (no `..`, no `/`, no newlines).
const AGENT_NAME = /^[a-z0-9-]+$/;

export interface RunOptions {
  // Agent name. Looked up at ~/agents/<name>/bin/<name>.
  name: string;
  // Optional initial prompt. When omitted, the launcher starts interactively.
  prompt?: string;
  // Optional per-call model override. Appended as `--model <X>` to the
  // launcher invocation; pi honors the later flag over the one baked
  // into the launcher.
  model?: string;
  // If true, omit `--print`/`-p` and let pi run interactively. Defaults
  // to non-interactive (--print) when a prompt is provided.
  interactive?: boolean;
  // If true, capture the agent's stdout instead of inheriting. Used by
  // the Discord listener to harvest a reply for auto-posting. Defaults
  // to false (stdio:inherit — output goes to the parent process).
  //
  // Caveat: incompatible with interactive=true. pi in interactive mode
  // expects to drive the user's terminal; capturing stdout breaks the
  // TUI. runAgent enforces this.
  captureStdout?: boolean;
  // Override the agents root dir (tests). Defaults to ~/agents.
  agentsRoot?: string;
  // Override child_process.spawn (tests).
  spawnFn?: SpawnFn;
}

export interface RunResult {
  exitCode: number;
  launcherPath: string;
  args: readonly string[];
  // Captured stdout, populated only when captureStdout=true. Undefined
  // otherwise (stdio:inherit means we didn't see the output).
  stdout?: string;
}

export async function runAgent(opts: RunOptions): Promise<RunResult> {
  if (!AGENT_NAME.test(opts.name)) {
    throw new Error(`invalid agent name: ${JSON.stringify(opts.name)} (must match ${AGENT_NAME})`);
  }
  if (opts.captureStdout && opts.interactive) {
    throw new Error(
      "runAgent: captureStdout is incompatible with interactive (interactive needs a live TTY)",
    );
  }
  const root = opts.agentsRoot ?? join(homedir(), "agents");
  const launcherPath = join(root, opts.name, "bin", opts.name);
  if (!existsSync(launcherPath)) {
    throw new Error(
      `bob run ${opts.name}: launcher not found at ${launcherPath} (run 'bob onboard ${opts.name}' first)`,
    );
  }

  const args: string[] = [];
  if (opts.model) {
    args.push("--model", opts.model);
  }
  // Default to --print (non-interactive) when prompt is provided AND user
  // didn't ask for interactive. With no prompt and not-interactive, the
  // launcher still starts an interactive session — pi's default behavior.
  if (opts.prompt !== undefined && opts.interactive !== true) {
    args.push("--print");
  }
  if (opts.prompt !== undefined) {
    args.push(opts.prompt);
  }

  const spawnFn = opts.spawnFn ?? (nodeSpawn as SpawnFn);
  // When capturing, pipe stdout but leave stdin + stderr inherited so the
  // user still sees errors and any interactive prompts (e.g., model auth)
  // route through normally. Otherwise inherit all three (existing behavior).
  const stdio: SpawnOptions["stdio"] = opts.captureStdout
    ? ["inherit", "pipe", "inherit"]
    : "inherit";

  const { exitCode, stdout } = await spawnAndWait(spawnFn, launcherPath, args, {
    stdio,
    env: process.env,
  });

  return {
    exitCode,
    launcherPath,
    args,
    ...(opts.captureStdout ? { stdout } : {}),
  };
}

function spawnAndWait(
  spawnFn: SpawnFn,
  command: string,
  args: readonly string[],
  options: SpawnOptions,
): Promise<{ exitCode: number; stdout: string }> {
  return new Promise((resolve, reject) => {
    const child = spawnFn(command, args, options);
    let stdout = "";
    if (child.stdout) {
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        stdout += chunk;
      });
    }
    child.on("error", reject);
    child.on("exit", (code) => resolve({ exitCode: code ?? 0, stdout }));
  });
}
