# Milton crate — tokenizer + GGUF dequant + native forward

Pure Rust nomic-embed-text-v1.5 WordPiece + Flair prefix convention, GGUF load
and dequant of the pinned Q4_K_M file, and the nomic-bert forward pass
(embeddings → transformer layers → mean-pool from the GGUF → L2). WASM
packaging is a later slice.

```rust
use milton::{tokenize, Prefix};

let ids = tokenize("hello", Prefix::Document);
// [CLS] search _ document : hello [SEP]
```

- **Tokenizer is pure:** `tokenize` / `apply_prefix` do no I/O.
- **Vocab is embedded** from the pinned `vocab/vocab.txt` (HF nomic-embed-text-v1.5 @ `e9b6763…`).
- **Prefixes** are byte-identical to `harness/lib/prefix.js`: `search_document: ` / `search_query: ` / passthrough. The space after the colon is load-bearing.

Goldens: `harness/goldens/tokens.json`. Pin + digest: `harness/goldens/tokenizer-pin.json`.

```sh
cargo test --manifest-path crate/Cargo.toml
cargo run --manifest-path crate/Cargo.toml --features cli --bin milton-tokenize -- --check-goldens
```

## GGUF load + dequant (issue #4)

Loads the pinned **nomic-embed-text-v1.5** GGUF and dequantizes its tensor types
to f32. Correctness oracle: llama.cpp `ggml_get_type_traits()->to_float` at the
commit in `harness/goldens/pin.json`. camelid GGUF/dequant patterns are mirrored
(MIT, github.com/timtoole02/Camelid), not vendored.

### What is in this GGUF (read from the file, not assumed)

| fact | value as stored |
|---|---|
| `general.architecture` | `nomic-bert` |
| `general.name` | `nomic-embed-text-v1.5` |
| `nomic-bert.block_count` | 12 |
| `nomic-bert.embedding_length` | 768 |
| `nomic-bert.context_length` | 2048 |
| `nomic-bert.pooling_type` | `1` → **mean** |
| `nomic-bert.attention.layer_norm_epsilon` | `1e-12` |
| embedding L2 normalize | **absent from GGUF** (harness applies `--embd-normalize 2` at embed time) |
| tensor types present | **F32, Q4_K, Q5_K, Q6_K** (Q4_K_M mix) |

Q8_0 and F16 are **not** in this file. The issue still requires those kernels;
their goldens are llama.cpp `from_float_ref` → `to_float` of a fixed ramp.

```sh
npm run dequant:goldens   # llama.cpp dump + package harness/goldens/dequant.json
cargo test --manifest-path crate/Cargo.toml
cargo run --manifest-path crate/Cargo.toml --bin dequant-gate
cargo run --manifest-path crate/Cargo.toml --bin dequant-must-fail
```

## Native forward (issue #5)

GGUF-driven nomic-bert: token + type-0 embeddings, post-norm layers, RoPE NEOX,
SwiGLU, mean pool (`nomic-bert.pooling_type = 1`), L2 (`--embd-normalize 2`).
Prefixes are `PrefixConfig`, not architecture code.

```sh
cargo run --manifest-path crate/Cargo.toml --release --bin embed-gate
cargo run --manifest-path crate/Cargo.toml --release --bin embed-must-fail
```

Never loosen `harness/goldens/epsilon.json` to pass. The oracle is llama.cpp.

Epsilon is derived the same way as the harness: run the reference twice,
measure the floor, set the gate a 10× margin above it with a numeric floor
(`harness/goldens/dequant-epsilon.json`). Do not loosen to pass.
