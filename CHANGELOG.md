# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-21

First publishable release. Bob is functional end-to-end: scaffold an agent with `bob onboard`, shape its persona through conversation, run it via `bob run`, keep it listening on Discord via `bob serve --discord`, and health-check it with `bob doctor`.

### Added

- **`bob onboard <name> --role <role>`** — scaffolds the per-agent directory layout (soul.md, bob.yaml, Ed25519 keypair, executable launcher, pi-agent config) and drops into an interactive hiring interview where the agent writes its own persona via conversation with the human. `--no-interactive` skips the interview for CI/dry runs. `--dry-run` previews the plan.
- **`bob align <name>`** — recurring drift-check counterpart. Spawns an alignment conversation; agent rewrites soul.md with the deltas surfaced.
- **`bob run <name> [prompt]`** — invoke an agent's launcher with optional `--model X` per-call override (lightweight dynamic routing) and `--interactive` for a TUI session.
- **`bob serve <name>`** — daemon mode. Mail consumer always-on. Add `--discord --discord-token-file <p> --discord-channels <ids> [--discord-dispatch-all] [--discord-model <m>]` to attach a Discord listener that dispatches @-mentions (or all messages in dispatchAll mode) through `runAgent` and posts the captured stdout back to Discord.
- **`bob doctor <name>`** — health check that walks the expected layout and reports per-check status (OK / WARN / FAIL / SKIP) with actionable `fix:` hints. Exit 1 on any FAIL. Read-only — never modifies state.
- **Five role templates** ship in `packages/shell/roles/`: `ea`, `writer`, `reviewer`, `coder`, `qa`, plus a blank-slate `custom` stub. Each role has a seed `soul.md` (What you own / Don't own / Personality / Tone / Failure modes) and a `role.json` with a sensible tool allowlist and provider/model defaults.
- **Launcher generation** writes a per-agent `~/.pi-agent/auth.json` (placeholder API key) and `models.json` (gateway baseUrl override) when provider is `exe-dev-gateway`, and appends `--append-system-prompt "$(cat $AGENT_DIR/soul.md)"` so the agent loads its persona on every invocation.
- **Identity** via Ed25519 keypair generation + Flair Agent record registration.
- **Discord bridge abstraction** (shell) + discord.js binding (separate `@tpsdev-ai/bob-discord` package so callers without Discord don't pay the ~30MB discord.js install cost).
- **stdout capture** in `runAgent` so the Discord listener can post the agent's reply.
- **Path-traversal + prompt-injection defense** at the entry of every agent-name-accepting function: `runOnboard`, `runAlign`, `runAgent`, `runDoctor`. AGENT_NAME = `/^[a-z0-9-]+$/`.

### Architecture

Bob is a thin TypeScript shell on top of [pi-coding-agent](https://github.com/earendil-works/pi). Pi owns the agent loop, tools, and LLM provider abstraction. Bob adds the office plumbing — identity, mailbox, channels, scheduling — that turns one terminal agent into a named office hire.

### Packages

- `@tpsdev-ai/bob` — the `bob` CLI command
- `@tpsdev-ai/bob-shell` — runtime + role templates + integrations (mail, Discord bridge abstraction, init, run, onboard/align, doctor)
- `@tpsdev-ai/bob-discord` — discord.js binding for `bob-shell`'s `DiscordClient` interface

### Status

`0.1.0` is the first publishable release. First production deployment is **Pulse-EA** on a fresh `tps-pulse` VM. The interactive onboard flow, real `bob run`, Discord listener with auto-reply, doctor, role templates, and per-agent pi config seeding all landed in PR-15 through PR-22.

Known caveats:
- Federation pair (Bob agents reading/writing Flair memory across hosts) requires the same setup as Reed Phase 2; not packaged into `bob office join` yet.
- Branch-office docs (`bob office join`) are stubbed pending the productization recipe from `~/ops/specs/`.

[0.1.0]: https://github.com/tpsdev-ai/bob/releases/tag/v0.1.0
