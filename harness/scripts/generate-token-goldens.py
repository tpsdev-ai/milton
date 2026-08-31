#!/usr/bin/env python3
"""Generate golden token-ID sequences from the pinned nomic HF tokenizer.

Oracle: HuggingFace `tokenizers` loading the committed tokenizer.json
(nomic-ai/nomic-embed-text-v1.5 @ e9b6763023c676ca8431644204f50c2b100d9aab).
Prefix bytes match harness/lib/prefix.js (Flair NOMIC_TEMPLATES).

This script is harness-only. It does not live in src/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "harness" / "corpus" / "corpus.json"
TOKENIZER_JSON = ROOT / "harness" / "goldens" / "tokenizer" / "tokenizer.json"
TOKENIZER_CONFIG = ROOT / "harness" / "goldens" / "tokenizer" / "tokenizer_config.json"
VOCAB_TXT = ROOT / "crate" / "vocab" / "vocab.txt"
OUT_TOKENS = ROOT / "harness" / "goldens" / "tokens.json"
OUT_PIN = ROOT / "harness" / "goldens" / "tokenizer-pin.json"

# Fail-closed pins — an unpinned reference is not a reference.
WANT = {
    "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
    "hf_revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
    "tokenizer_json_sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config_sha256": "d7e0000bcc80134debd2222220427e6bf5fa20a669f40a0d0d1409cc18e0a9bc",
    "vocab_txt_sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}

# Byte-identical to harness/lib/prefix.js (space after colon is load-bearing).
PREFIX = {
    "document": "search_document: ",
    "query": "search_query: ",
    "none": "",
}


def sha256file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def canonical_json(value) -> str:
    def sort_deep(v):
        if isinstance(v, list):
            return [sort_deep(x) for x in v]
        if isinstance(v, dict):
            return {k: sort_deep(v[k]) for k in sorted(v)}
        return v

    # Match harness/lib/digest.js canonicalJson (JSON.stringify, UTF-8, no ASCII escapes).
    return json.dumps(sort_deep(value), separators=(",", ":"), ensure_ascii=False)


def sha256json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def apply_prefix(text: str, kind: str) -> str:
    if kind not in PREFIX:
        raise SystemExit(f"fail-closed: invalid prefix kind {kind!r}")
    if not isinstance(text, str):
        raise SystemExit(f"fail-closed: text must be a string, got {type(text)}")
    return PREFIX[kind] + text


def main() -> int:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("fail-closed: python tokenizers package missing — pip install tokenizers==0.21.0", file=sys.stderr)
        return 1

    import tokenizers

    got = {
        "tokenizer_json": sha256file(TOKENIZER_JSON),
        "tokenizer_config": sha256file(TOKENIZER_CONFIG),
        "vocab_txt": sha256file(VOCAB_TXT),
    }
    want_map = {
        "tokenizer_json": WANT["tokenizer_json_sha256"],
        "tokenizer_config": WANT["tokenizer_config_sha256"],
        "vocab_txt": WANT["vocab_txt_sha256"],
    }
    for name, digest in got.items():
        if digest != want_map[name]:
            print(f"fail-closed: {name} digest {digest} != pin {want_map[name]}", file=sys.stderr)
            return 1

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if corpus.get("schema") != "milton.corpus/1":
        print("fail-closed: unexpected corpus schema", file=sys.stderr)
        return 1

    corpus_digest = sha256json(
        [{"id": c["id"], "prefix": c["prefix"], "text": c["text"]} for c in corpus["cases"]]
    )

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    items = []
    for case in corpus["cases"]:
        kind = case["prefix"]
        prefixed = apply_prefix(case["text"], kind)
        enc = tok.encode(prefixed)
        items.append(
            {
                "id": case["id"],
                "prefix": kind,
                "prefixed": prefixed,
                "n_ids": len(enc.ids),
                "ids": list(enc.ids),
                "tokens": list(enc.tokens),
            }
        )

    tokens_digest = sha256json([{"id": it["id"], "ids": it["ids"], "prefix": it["prefix"]} for it in items])

    nfc = next(it for it in items if it["id"] == "unicode-nfc")
    nfd = next(it for it in items if it["id"] == "unicode-nfd")
    nfc_nfd_ids_equal = nfc["ids"] == nfd["ids"]

    pin = {
        "schema": "milton.tokenizer-pin/1",
        "model": "nomic-embed-text-v1.5",
        "tokenizer_class": "BertTokenizer",
        "source": {
            "kind": "huggingface",
            "repo": WANT["hf_repo"],
            "revision": WANT["hf_revision"],
            "files": {
                "tokenizer.json": {
                    "path": "harness/goldens/tokenizer/tokenizer.json",
                    "sha256": WANT["tokenizer_json_sha256"],
                    "bytes": TOKENIZER_JSON.stat().st_size,
                },
                "tokenizer_config.json": {
                    "path": "harness/goldens/tokenizer/tokenizer_config.json",
                    "sha256": WANT["tokenizer_config_sha256"],
                    "bytes": TOKENIZER_CONFIG.stat().st_size,
                },
                "vocab.txt": {
                    "path": "crate/vocab/vocab.txt",
                    "sha256": WANT["vocab_txt_sha256"],
                    "bytes": VOCAB_TXT.stat().st_size,
                    "n_tokens": 30522,
                },
            },
            "url": f"https://huggingface.co/{WANT['hf_repo']}/tree/{WANT['hf_revision']}",
        },
        "oracle": {
            "package": "tokenizers",
            "package_version": tokenizers.__version__,
            "loader": "Tokenizer.from_file(tokenizer.json)",
            "note": "Official HF rust tokenizers binding loading nomic's committed tokenizer.json (BertNormalizer + BertPreTokenizer + WordPiece + [CLS]/[SEP]).",
        },
        "prefix_convention": {
            "document": "search_document: {text}",
            "query": "search_query: {text}",
            "none": "{text}",
            "source": "flair resources/embeddings-provider.ts + harper-fabric-embeddings src/engine.ts NOMIC_TEMPLATES; replicated in harness/lib/prefix.js",
        },
        "pipeline": {
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": True,
                "strip_accents": None,
                "lowercase": True,
                "strip_accents_effective": True,
            },
            "pre_tokenizer": "BertPreTokenizer",
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
            },
            "post_processor": "[CLS] $A [SEP]",
            "special_ids": {"PAD": 0, "UNK": 100, "CLS": 101, "SEP": 102, "MASK": 103},
        },
        "corpus_digest": corpus_digest,
        "tokens_digest": tokens_digest,
        "n_cases": len(items),
        "observed": {
            "nfc_nfd_ids_equal": nfc_nfd_ids_equal,
            "nfc_nfd_note": (
                "BERT strip_accents (NFD + drop Mn) collapses unicode-nfc and unicode-nfd "
                "to the same token IDs. The corpus trap prose anticipated they would not "
                "collide; the pinned HF nomic tokenizer does. Goldens record the reference, "
                "not the trap guess. A pre-tokenize Unicode NFC fold is still a distinct "
                "bug class (it is not what this oracle does as a standalone step — accent "
                "strip is the BERT normalizer's own NFD)."
            ),
        },
    }

    goldens = {
        "schema": "milton.token-goldens/1",
        "model": "nomic-embed-text-v1.5",
        "corpus_digest": corpus_digest,
        "tokens_digest": tokens_digest,
        "tokenizer_pin": "harness/goldens/tokenizer-pin.json",
        "n": len(items),
        "items": items,
    }

    OUT_PIN.write_text(json.dumps(pin, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    OUT_TOKENS.write_text(json.dumps(goldens, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": [str(OUT_PIN.relative_to(ROOT)), str(OUT_TOKENS.relative_to(ROOT))],
        "n": len(items),
        "corpus_digest": corpus_digest,
        "tokens_digest": tokens_digest,
        "oracle": f"tokenizers=={tokenizers.__version__}",
        "hf_revision": WANT["hf_revision"],
        "nfc_nfd_ids_equal": nfc_nfd_ids_equal,
        "per_case": {it["id"]: it["n_ids"] for it in items},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
