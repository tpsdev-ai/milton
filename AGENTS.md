# Instructions for agents working in this repo

This repo is built primarily by remote agents (Grok / Cursor Cloud / bot lanes) with Flint + K&S oversight. Read this fully before any task.

## What this is

Milton — an in-process Node.js embeddings module for Flair (see README). The whole point is **correctness proven against a pinned reference**, and **light** (no native runtime dragged into the install). Work is tracked in GitHub issues; each is self-contained.

## The one rule that overrides everything: the golden-vector gate is the definition of done

An embedding is correct **only** when it matches the pinned reference (nomic-embed-text-v1.5 via llama.cpp) within the agreed epsilon, over the conformance corpus. Not "looks reasonable." Not "the tests I wrote pass." The reference vectors are the oracle. If your implementation doesn't match them, it is wrong — say so and keep iterating; do not loosen the epsilon to pass. A subtly-wrong embedder is the worst possible outcome here (silent recall collapse downstream), so **fail closed**: an unverified path must refuse, never guess.

## Hard constraints

1. **Public repo. No secrets, ever** — no keys/tokens/internal hostnames in code, config, commits, issues, or PR text. Config is env-driven with a placeholder-only `.env.example`.
2. **Light by construction.** The reason Milton exists is footprint. Do not add a heavy native runtime (onnxruntime-node, a full llama.cpp binding used *at runtime*) to the shipped package — those are allowed only in `harness/` as the *reference oracle*, never in `src/`. If you think you need one in `src/`, STOP and raise it as a finding.
3. **One PR per issue, small PRs, `Refs #N`.** Do not merge your own PRs; do not push to main.
4. **Every PR body states how you verified** — the exact commands and their output (including the conformance run: N vectors compared, max/mean epsilon, pass/fail), or the words "not verified" with why. A claim without a command is noise.
5. **Scope discipline:** v1 is nomic-embed-text-v1.5 only. Do not generalize to other architectures speculatively.

## Authority to refuse

If an issue's approach is wrong — impossible within the footprint budget, a reference-mismatch you can't close, a primitive that's missing — STOP and say so in a comment. A blocked report with the actual mismatch (expected vs got, epsilon) is a successful outcome. Do not force it through or quietly widen the tolerance.

## Environment

`.cursor/environment.json` / build config sets up Node + the reference toolchain (llama.cpp for generating golden vectors lives in `harness/`, not shipped). Never run against any pre-existing `~/.flair`.
