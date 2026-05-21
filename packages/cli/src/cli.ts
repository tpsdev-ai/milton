#!/usr/bin/env node
// Bob CLI — bob <subcommand> [args]
//
// PR-1 ships the surface stubs. Each subcommand prints what it WILL do
// in PR-2+ and exits 0. This is intentional: it gates K&S review on
// the type surface + role-template structure before we hand-roll the
// runtime.

import { MailConsumer, initAgent, loadRole, type BobRole } from "@tpsdev-ai/bob-shell";

interface Args {
  command: string;
  positional: string[];
  flags: Record<string, string | boolean>;
}

function parseArgs(argv: string[]): Args {
  const [command = "help", ...rest] = argv;
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i];
    if (tok.startsWith("--")) {
      const key = tok.slice(2);
      const next = rest[i + 1];
      if (!next || next.startsWith("--")) {
        flags[key] = true;
      } else {
        flags[key] = next;
        i++;
      }
    } else {
      positional.push(tok);
    }
  }
  return { command, positional, flags };
}

function help(): void {
  console.log(`Bob — moldable office-agent shell.

Usage: bob <command> [args]

Commands:
  init <name>         Bootstrap a new Bob-shaped agent
                      Flags: --role <r> --provider <p> --model <m>
                             --dry-run --force
  run <name> [prompt] Run one session for the named agent
  serve <name>        Run mail watcher + cron loop as a daemon
  doctor <name>       Health check (identity, mail, channels, provider auth)
  office join <name>  Join an existing branch office
  help                Show this help

Roles: ea | writer | reviewer | coder | qa | custom`);
}

function init(name: string, flags: Record<string, string | boolean>): void {
  const role = (flags.role ?? "custom") as BobRole;
  const provider = String(flags.provider ?? "ollama-cloud");
  const model = String(flags.model ?? "kimi-k2.6");
  const dryRun = flags["dry-run"] === true;
  const force = flags.force === true;

  if (dryRun) {
    // Validate role exists, print plan, exit without writing.
    const template = loadRole(role);
    console.log(`[bob init] PLAN (--dry-run):
  agent.id        = ${name}
  agent.role      = ${role}
  provider.name   = ${provider}
  provider.model  = ${model}
  soul (from template, ${template.soul.length} chars) → ~/agents/${name}/soul.md
  tools.allow     = ${template.tools.allow.join(", ")}
  bin/launcher    → ~/agents/${name}/bin/${name}
  bob.yaml     → ~/agents/${name}/bob.yaml`);
    return;
  }

  const result = initAgent({ name, role, provider, model, noClobber: !force });
  console.log(`[bob init] wrote ${result.files.length} files to ${result.agentDir}`);
  for (const f of result.files) console.log(`  ${f}`);
  console.log(`\nNext: chmod-verified launcher is at ${result.agentDir}/bin/${name}.`);
  console.log(`Edit ${result.agentDir}/bob.yaml or ${result.agentDir}/soul.md to customize.`);
  console.log(`Flair pair + TPS mail inbox are PR-3 work — not wired yet.`);
}

function run(name: string, prompt?: string): void {
  console.log(`[bob run] PR-1 stub — would invoke pi-coding-agent for ${name}${prompt ? ` with prompt: ${prompt}` : " (interactive)"}`);
}

function serve(name: string): void {
  const consumer = new MailConsumer({ name });
  consumer.start();
  console.log(`[bob serve] mail consumer running for ${name}`);
  console.log(`  inbox:    ~/.tps/mail/${name}/`);
  console.log(`  launcher: ~/agents/${name}/bin/${name}`);
  console.log(`  poll:     2s · processed:0 failed:0`);
  console.log(`Ctrl-C to stop. (Cron + Discord wire in PR-5+.)`);
  // Keep the event loop alive
  const shutdown = () => {
    console.log(`\n[bob serve] stopping; stats: ${JSON.stringify(consumer.stats)}`);
    consumer.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

function doctor(name: string): void {
  console.log(`[bob doctor] PR-1 stub — would check identity, mail, channels, provider auth for ${name}`);
}

function main(): number {
  const args = parseArgs(process.argv.slice(2));
  try {
    switch (args.command) {
      case "init": {
        const name = args.positional[0];
        if (!name) { console.error("bob init: missing <name>"); return 2; }
        init(name, args.flags);
        return 0;
      }
      case "run":
        if (!args.positional[0]) { console.error("bob run: missing <name>"); return 2; }
        run(args.positional[0], args.positional.slice(1).join(" ") || undefined);
        return 0;
      case "serve":
        if (!args.positional[0]) { console.error("bob serve: missing <name>"); return 2; }
        serve(args.positional[0]);
        return 0;
      case "doctor":
        if (!args.positional[0]) { console.error("bob doctor: missing <name>"); return 2; }
        doctor(args.positional[0]);
        return 0;
      case "help":
      case "--help":
      case "-h":
        help();
        return 0;
      default:
        console.error(`bob: unknown command '${args.command}'. Run 'bob help'.`);
        return 2;
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`bob: ${msg}`);
    return 1;
  }
}

process.exit(main());
