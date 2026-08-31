# Milton

> *"Excuse me, I believe you have my stapler."*

**The in-process Node.js embeddings module for [Flair](https://github.com/tpsdev-ai/flair).** One precise thing — text → vector — done flawlessly, in-process, with no separate binary and no native runtime to install. Every embedding is verified against a pinned reference before it ships. The cover sheet that has to be *right*, or the whole report bounces.

## Why Milton exists

Flair's memory is only as good as its embeddings, and today they come through a heavy inference runtime. Milton is the featherweight alternative: `npm i` and you have embeddings, in-process, no sidecar server, no GGUF-runtime dependency dragged into the install. It does the one job the overlooked, load-bearing worker always did — quietly, exactly, and without which nothing downstream works.

## The non-negotiable: correctness is a gate, not a hope

Milton is **evidence-gated**. Its output is defined as correct only when it matches a pinned reference implementation (nomic-embed-text-v1.5 via llama.cpp — the v1 oracle) within epsilon, over a conformance corpus — token-for-token discipline applied to vectors. An implementation that produces plausible-but-wrong vectors is the worst failure a memory layer can have: silent recall collapse, no error, nobody notices for weeks. So Milton **fails closed on any unverified path** — an embedding that hasn't cleared the golden-vector gate does not ship. (Reference discipline modeled on [camelid](https://github.com/timtoole02/Camelid).)

## Scope (v1)

- **GGUF-driven, nomic verified-first.** Dequant and forward read architecture from the file — layer count, dims, pooling, normalization. We ship nomic-embed-text-v1.5 for v1 because Flair uses it today; that file is the golden-vector oracle, not a hardwired product. Prefix convention (`search_query:` / `search_document:`) is config, not code. BERT-family (bge / gte / e5 / nomic-v2) is the same shape parameterized by the GGUF. Model #2 is pin goldens + prefix config + any GGUF-flagged arch variant (CLS vs mean), not a rewrite.
- **In-process, Node.** No separate server, no OpenAI-compatible HTTP hop — a library you call.
- **Next file, not a rewrite.** A second BERT-family GGUF is goldens + prefix config + whatever pooling the file flags. No model registry.

## Status

Phase 1 (issue #1): the conformance + bench harness lives in `harness/`. There is no embedder yet — `src/` refuses. See `harness/README.md` and `docs/HARNESS-SPEC.md`.

## Layout (planned)

```
harness/   the golden-vector conformance + footprint/throughput bench (built first)
src/       the embedder (tokenizer + forward pass)
docs/      HARNESS-SPEC.md, architecture, receipts
```
